import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from contextlib import contextmanager
import torch.nn as nn

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

from bert_score import BERTScorer

# from vilmedic of radvlm eval
class BertScore(nn.Module):
    def __init__(self):
        super(BertScore, self).__init__()
        with torch.no_grad():
            self.bert_scorer = BERTScorer(model_type='distilbert-base-uncased',
                                          num_layers=5,
                                          batch_size=64,
                                          nthreads=4,
                                          all_layers=False,
                                          idf=False,
                                          device=None,
                                          lang='en',
                                          rescale_with_baseline=True,
                                          baseline_path=None)

    def forward(self, refs, hyps):
        p, r, f = self.bert_scorer.score(
            cands=hyps,
            refs=refs,
            verbose=False,
            batch_size=64,
        )
        return torch.mean(f).item(), f.tolist()

app = FastAPI()
scorer = BertScore()
sem = asyncio.Semaphore(1)

class ScoreRequest(BaseModel):
    pred_text: str
    true_text: str

class ScoreResponse(BaseModel):
    score: float


@app.post("/score_bertscore", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    async with sem:
        with mem_slim():
            print(f"{len(request.pred_text)=}")
            print(f"{len(request.true_text)=}")
            if len(request.pred_text) <= 5 or len(request.true_text) <= 5:
                print("too short return 0.0.")
                return ScoreResponse(score=0.0)
            
            try:
                # Just take first 2048 characters to score if too long
                _, rewards = scorer(hyps=[request.pred_text[:2048].lower()], refs=[request.true_text[:2048].lower()])
            except Exception as e:
                print(f"BertScore scoring failed: {e}")
                return ScoreResponse(score=0.0)
            
        print(rewards)
        bert_score = rewards[0]
        return ScoreResponse(score=bert_score)
