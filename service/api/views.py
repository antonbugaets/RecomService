from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from service.api.exceptions import (
    InvalidApiKey,
    MissingApiKey,
    UserNotFoundError,
    WrongModelNameError,
)
from service.log import app_logger
from service.models import Error
from service.settings import ServiceConfig, get_config


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def get_api_key(
    api_key: Optional[str] = Security(api_key_header),
    service_config: ServiceConfig = Depends(get_config),
) -> str:
    if api_key is None:
        raise MissingApiKey()
    if api_key == service_config.api_key:
        return api_key
    raise InvalidApiKey()


router = APIRouter()


unauthorized_examples = {
    "Missing api key": {
        "value": [
            {
                "error_key": "missing_api_key",
                "error_message": "Missing header with api key",
                "error_loc": None,
            },
        ],
    },
    "Invalid api key": {
        "value": [
            {
                "error_key": "invalid_api_key",
                "error_message": "Invalid api key",
                "error_loc": None,
            },
        ],
    },
}

not_found_examples = {
    "Unknown user": {
        "value": [
            {
                "error_key": "user_not_found",
                "error_message": "User is unknown",
                "error_loc": None,
            },
        ],
    },
    "Wrong model name": {
        "value": [
            {
                "error_key": "wrong_model_name",
                "error_message": "Wrong model name",
                "error_loc": None,
            },
        ],
    },
}

recommendations_example = {
  "user_id": 0,
  "items": list(range(10)),
}

unauthorized_response = {
    "model": List[Error],
    "content": {
        "application/json": {
            "examples": unauthorized_examples,
        },
    },
}

not_found_response = {
    "model": List[Error],
    "content": {
        "application/json": {
            "examples": not_found_examples,
        },
    },
}

recommendations_response = {
    "description": "Recommendations for user",
    "content": {
        "application/json": {
            "example": recommendations_example,
        },
    },
}


@router.get(
    path="/health",
    tags=["Health"],
    responses={401: unauthorized_response},
)
async def health(api_key: str = Depends(get_api_key)) -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        200: recommendations_response,
        401: unauthorized_response,
        404: not_found_response,
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    api_key: str = Depends(get_api_key),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs
    if model_name == "test_model":
        reco = list(range(k_recs))
    elif model_name in request.app.state.models:
        model = request.app.state.models[model_name]
        reco = model.predict(user_id, k_recs=k_recs)
    else:
        raise WrongModelNameError(
            error_message=f"Wrong model name — '{model_name}'",
        )
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
