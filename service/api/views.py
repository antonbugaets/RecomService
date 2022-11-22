from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from service.api.exceptions import UserNotFoundError, WrongModelNameError
from service.log import app_logger
from service.models import Error


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


router = APIRouter()


unauthorized_examples = {
    "Missing api key header": {
        "value": [
            {
                "error_key": "http_exception",
                "error_message": "Not authenticated",
                "error_loc": None,
            },
        ],
    },
    "Wrong api key": {
        "value": [
            {
                "error_key": "http_exception",
                "error_message": "Wrong api key",
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


@router.get(
    path="/health",
    tags=["Health"],
    responses={
        401: {
            "model": List[Error],
            "content": {
                "application/json": {
                    "examples": unauthorized_examples,
                },
            },
        },
    },
)
async def health(api_key: str = Depends(oauth2_scheme)) -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        200: {
            "description": "Recommendations for user",
            "content": {
                "application/json": {
                    "example": recommendations_example,
                },
            },
        },
        401: {
            "model": List[Error],
            "content": {
                "application/json": {
                    "examples": unauthorized_examples,
                },
            },
        },
        404: {
            "model": List[Error],
            "content": {
                "application/json": {
                    "examples": not_found_examples,
                },
            },
        },
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    api_key: str = Depends(oauth2_scheme),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    # Write your code here

    if api_key != request.app.state.api_key:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="Wrong api key",
        )

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs
    if model_name == 'test_model':
        reco = list(range(k_recs))
    else:
        raise WrongModelNameError(
            error_message=f"Wrong model name — '{model_name}'",
        )
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
