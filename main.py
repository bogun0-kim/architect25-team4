import asyncio
import time
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from managers.llm_manager import LLM
from managers.tool_manager import ToolManager
from managers.prompt_manager import PromptManager
from conductor import build
from _demo import prepare
import agents


prepare(agents)
# TODO : make thread_id logic by user_id
config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}

# Conductor build
start_time = time.time()
conductor = build(LLM.get(), ToolManager.data(), PromptManager.get(LLM.name()), with_preplan=False)
print(f'# Built conductor ({time.time() - start_time:.3f} seconds)')


async def generate_response(user_message: str) -> AsyncGenerator[bytes, None]:
    user_message = user_message.strip()
    states = list(conductor.get_state_history(config))
    user_request: str = None
    last_message: BaseMessage = None

    if states and states[0].values:
        user_request = states[0].values["user_request"]
        last_message = states[0].values["messages"][-1]

    if isinstance(last_message, AIMessage) or last_message is None:
        print("Last message is from the AI or no messages found. Starting a new conversation.")
        config["configurable"]["thread_id"] = uuid.uuid4()
        user_request = user_message
        messages = [HumanMessage(content=user_message)]
    elif isinstance(last_message, SystemMessage) and "[HumanInTheLoop]" in last_message.content:
        print("Last message indicates human input is needed. Continuing the conversation.")
        messages = states[0].values["messages"] + [HumanMessage(content=user_message)]
    else:
        print("Last message is from the user or an unexpected type. Starting a new conversation.")
        config["configurable"]["thread_id"] = uuid.uuid4()
        user_request = user_message
        messages = [HumanMessage(content=user_message)]

    start_time = time.time()
    print('\n########## START ##########\n')
    n_steps = 0
    done = True
    yield '<< Processing >>'
    await asyncio.sleep(0.5)
    for step in conductor.stream({
        "user_request": user_request,
        "messages": messages,
    }, config=config):
        n_steps += 1
        step_name = list(step)[0]
        print(f'\n#### [STEP-{n_steps}-{step_name}] ####')
        if step_name == "__interrupt__":
            print(f'# [Interrupt] {step[step_name][0]}')
            done = False
            yield str(step[step_name][0].value.strip()).encode('utf-8')
        else:
            messages = step[step_name]["messages"]
            for i, msg in enumerate(messages):
                print(f'# [message-{i}] {msg}')
                if not isinstance(msg, (AIMessage, ToolMessage)):
                    continue
                content = msg.content.replace('[HumanInTheLoop]', '').strip()
                if content == '':
                    continue
                if content.replace('join', '').strip() == '':
                    continue
                yield str(content).encode('utf-8')
        await asyncio.sleep(0.5)
    if done:
        yield '<< Done >>'
    print(f'\n########## DONE ({time.time() - start_time:.3f} seconds) ##########\n')


app = FastAPI()


@app.post('/test')
async def test(request: Request):
    data = await request.json()
    return StreamingResponse(generate_response(data.get("message", '')), media_type='text/plain')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
