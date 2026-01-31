import redis
from rq import Worker, Queue

from .config import SETTINGS
from .db import init_db


def main() -> None:
    init_db()
    redis_conn = redis.Redis.from_url(SETTINGS.redis_url)
    worker = Worker([Queue("tabscore", connection=redis_conn)], connection=redis_conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
