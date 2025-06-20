import os
import asyncio
import uuid
import time
import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from managers.llm_manager import LLM
from managers.tool_manager import ToolManager
from managers.prompt_manager import PromptManager
from conductor import build
from _demo import prepare
import agents


def collect_stream_sync(app, *arg, **kwargs):
    async def runner():
        outputs = []
        max_steps = 3
        async for output in app.astream(*arg, **kwargs):
            outputs.append(output)
            if len(outputs) >= max_steps:
                break
            await asyncio.sleep(0.5)
        await asyncio.sleep(0.5)
        return outputs
    return asyncio.run(runner())


def main(data_path, result_path):
    prepare(agents)
    conductor = build(LLM.get(), ToolManager.data(), PromptManager.get(LLM.name()), with_preplan=False)

    with open(data_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    done_indices = []
    if os.path.isfile(result_path):
        with open(result_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
        done_indices = [int(line.split(' ')[0].replace('[', '').replace(']', '').strip()) for line in lines]
    ____skip = [13, 20, 26, 29, 31, 33, 37, 45, 57, 58, 64, 68, 94, 97]

    for i, d in enumerate(data):
        if i in done_indices:
            print(f'[{i}] done')
            continue
        # if i in ____skip:
        #     print(f'[{i}] SKIP, TODO!!')
        #     continue

        start_time = time.time()
        x = d["query"]
        y_true = sorted(list(set(d["agents"])))
        print(f'[{i}] query="{x}", ground-truth={y_true}')

        try:
            config = {"configurable": {"thread_id": uuid.uuid4()}}
            outputs = collect_stream_sync(
                conductor,
                {"user_request": x, "messages": [HumanMessage(content=x)]},
                config=config)
        except Exception as e:
            print(f'[{i}] ERROR, predicted=N/A ({time.time() - start_time:.3f}s)\n{e}\n')
            continue

        y_pred = []
        for step in outputs:
            step_name = list(step)[0]
            if step_name == "__interrupt__":
                continue
            for msg in step[step_name]["messages"]:
                tool_name = msg.additional_kwargs.get("tool_calls", [{}])[0].get("function", {}).get("name", None)
                if tool_name is not None and tool_name != 'join':
                    y_pred.append(tool_name)
                if step_name == "join":
                    with open(result_path, 'a', encoding='utf8') as f:
                        f.write(f'{msg.content}\n')
            
        y_pred = sorted(list(set(y_pred)))

        log = f'[{i}] {y_true == y_pred}, GT={y_true}, PD={y_pred}, Query="{x}" ({time.time() - start_time:.3f}s)'
        with open(result_path, 'a', encoding='utf8') as f:
            f.write(f'{log}\n')
            f.write(f'###############################################\n')
        print(log)
        time.sleep(1.0)


def accuracy(result_path):
    with open(result_path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    corrects = []
    for line in lines:
        index, correct = line.split(' ')[:2]
        # index = int(index.replace('[', '').replace(']', '').strip())
        correct = correct.replace(',', '').strip() == 'True'
        corrects.append(1 if correct else 0)
    acc = sum(corrects) / len(corrects) * 100.0
    acc_log = f'# Accuracy: {acc:.2f} % ({sum(corrects)} / {len(corrects)})'
    with open(result_path, 'a', encoding='utf8') as f:
        f.write(f'{acc_log}\n')


if __name__ == '__main__':
    main('./_demo/data-eval-20250616.json', './result-250619-3.txt')
    accuracy('./result-250619-3.txt')
