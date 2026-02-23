from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend.app.ml_pipeline import (
    get_available_models,
    get_saved_metrics,
    predict_text,
    train_and_compare,
    train_single,
)
from backend.app.youtube_scraper import scrape_youtube


app = FastAPI(title="MCH Hate Speech API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str
    algorithm: str = "best"


class PredictResponse(BaseModel):
    label: str
    confidence: float
    algorithm: str


class TrainRequest(BaseModel):
    tune: bool = False
    fast: bool = False


class TrainResponse(BaseModel):
    best_algorithm: str
    dataset_size: int
    train_size: int
    validation_size: int
    metrics: dict
    artifacts: dict


class ScrapeRequest(BaseModel):
    url: str
    mode: str = "auto"
    max_items: int = 100
    predict: bool = True
    algorithm: str = "best"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = predict_text(payload.text, algorithm=payload.algorithm)
        return PredictResponse(**result)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {err}") from err


@app.post("/train", response_model=TrainResponse)
def train(payload: TrainRequest) -> TrainResponse:
    try:
        metadata = train_and_compare(tune=payload.tune, fast=payload.fast)
        return TrainResponse(**metadata)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Training failed: {err}") from err


@app.post("/train/rf", response_model=TrainResponse)
def train_rf(payload: TrainRequest) -> TrainResponse:
    try:
        metadata = train_single("random_forest", tune=payload.tune, fast=payload.fast)
        return TrainResponse(**metadata)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Training failed: {err}") from err


@app.post("/train/xgboost", response_model=TrainResponse)
def train_xgboost(payload: TrainRequest) -> TrainResponse:
    try:
        metadata = train_single("xgboost", tune=payload.tune, fast=payload.fast)
        return TrainResponse(**metadata)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Training failed: {err}") from err


@app.get("/metrics")
def metrics() -> dict:
    try:
        return get_saved_metrics()
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Load metrics failed: {err}") from err


@app.get("/models")
def models() -> dict:
    return {"available": get_available_models()}


@app.post("/scrape")
def scrape(payload: ScrapeRequest) -> dict:
    try:
        source, comments = scrape_youtube(
            url=payload.url,
            mode=payload.mode,
            max_items=max(1, min(payload.max_items, 500)),
        )

        rows = [{"text": text} for text in comments]
        prediction_enabled = payload.predict
        prediction_error = None

        if prediction_enabled and rows:
            try:
                for row in rows:
                    result = predict_text(row["text"], algorithm=payload.algorithm)
                    row.update(
                        {
                            "label": result["label"],
                            "confidence": result["confidence"],
                            "algorithm": result["algorithm"],
                        }
                    )
            except Exception as err:
                prediction_enabled = False
                prediction_error = str(err)

        summary = {
            "total_comments": len(rows),
            "hate_speech_count": sum(1 for row in rows if row.get("label") == "hate_speech"),
            "non_hate_speech_count": sum(1 for row in rows if row.get("label") == "non_hate_speech"),
        }

        return {
            "source": source,
            "prediction_enabled": prediction_enabled,
            "prediction_error": prediction_error,
            "summary": summary,
            "items": rows,
        }
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {err}") from err
