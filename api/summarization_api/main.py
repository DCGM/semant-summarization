import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from summarization_api.config import config
from summarization_api.routes import summary_route


tags_metadata = [
    {
        "name": "Summary",
        "description": "",
    },
]

app = FastAPI(openapi_tags=tags_metadata)


async def init_db():
    pass


@app.on_event("startup")
async def startup():
    await init_db()


app.include_router(summary_route, prefix="/api/summary")


#@app.exception_handler(DBError)
#async def unicorn_exception_handler(request: Request, exc: DBError):
#    return JSONResponse(status_code=400, content={"message": str(exc)})


if config.PRODUCTION:
    logging.warning(f'PRODUCTION')
else:
    logging.warning(f'DEVELOPMENT')
    app.add_middleware(
         CORSMiddleware,
         allow_origins=["http://localhost:9001", "http://127.0.0.1:9001", "http://pchradis2.fit.vutbr.cz:9001"],
         allow_credentials=True,
         allow_methods=["*"],
         allow_headers=["*"],
     )
