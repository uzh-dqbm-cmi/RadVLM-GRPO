import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from contextlib import contextmanager
import torch.nn as nn
from radgraph import F1RadGraph

torch.set_num_threads(1)
torch.cuda.empty_cache()

torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

@contextmanager
def mem_slim():
    #  torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        yield

app = FastAPI()
scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl")
sem = asyncio.Semaphore(1)

class ScoreRequest(BaseModel):
    pred_text: str
    true_text: str

class ScoreResponse(BaseModel):
    score: float


@app.post("/score_radgraph", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    async with sem:
        with mem_slim():
            print(f"{len(request.pred_text)=}")
            print(f"{len(request.true_text)=}")
            if len(request.pred_text) <= 5 or len(request.true_text) <= 5:
                print("too short return -0.0.")
                return ScoreResponse(score=-0.0)
            
            try:
                # Just take first 2048 characters to score if too long∂
                simple, partial, complete = scorer(hyps=[request.pred_text[:2048].lower()], refs=[request.true_text[:2048].lower()])[0]
                # simple, partial, complete
            except Exception as e:
                print(f"RadGraph scoring failed: {e}")
                return ScoreResponse(score=0.0)
            
        radgraph_score = complete
        return ScoreResponse(score=radgraph_score)
