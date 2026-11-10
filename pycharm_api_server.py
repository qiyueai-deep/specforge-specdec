"""PyCharm one-click entry point for the bounded-concurrency API service.

Run this file with the project root as PyCharm's working directory.  The
service defaults to two simultaneous generation requests, a 30-second queue
timeout, and one PyTorch CPU thread per request.
"""

from specforge.api import main


if __name__ == "__main__":
    main()
