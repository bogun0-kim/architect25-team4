import asyncio
import time
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from langchain_core.messages import HumanMessage
from managers.llm_manager import LLM
from managers.tool_manager import ToolManager
from managers.prompt_manager import PromptManager
from conductor import build
from _demo import prepare


prepare()
# conductor = build(LLM.get(), ToolManager.data(), PromptManager.get(LLM.name()))


async def generate_response(user_message: str) -> AsyncGenerator[bytes, None]:
    start_time = time.time()
    conductor = build(LLM.get(), ToolManager.data(), PromptManager.get(LLM.name()))
    print(f'# Built conductor ({time.time() - start_time:.3f} seconds)')

    start_time = time.time()
    print('\n########## START ##########\n')
    n_steps = 0
    yield '<< Processing >>'
    await asyncio.sleep(0.5)
    for step in conductor.stream({"messages": [HumanMessage(content=user_message)]}):
        n_steps += 1
        step_name = list(step)[0]
        messages = step[step_name]["messages"]
        print(f'\n#### [STEP-{n_steps}-{step_name}] ####')
        for i, msg in enumerate(messages):
            print(f'# [message-{i}] {msg}')
        yield str(messages).encode('utf-8')
        await asyncio.sleep(0.5)
    yield '<< Done >>'
    print(f'\n########## DONE ({time.time() - start_time:.3f} seconds) ##########\n')


app = FastAPI()


@app.post('/test')
async def test(request: Request):
    data = await request.json()
    return StreamingResponse(generate_response(data.get("message", '')), media_type='text/plain')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3000)
