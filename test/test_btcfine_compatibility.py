import unittest

from lighter.rest import RESTClientObject
from lighter.ws_client import WsClient


class _Response:
    status = 200
    reason = "OK"
    headers = {}

    async def read(self):
        return b""


class _PoolManager:
    def __init__(self):
        self.request_args = None

    async def request(self, **kwargs):
        self.request_args = kwargs
        return _Response()


class TestBtcfineCompatibility(unittest.IsolatedAsyncioTestCase):
    async def test_default_rest_timeout_is_thirty_seconds(self):
        pool_manager = _PoolManager()
        client = object.__new__(RESTClientObject)
        client.proxy = None
        client.proxy_headers = None
        client.pool_manager = pool_manager
        client.retry_client = None

        await client.request("GET", "https://example.invalid")

        self.assertEqual(pool_manager.request_args["timeout"], 30)

    def test_unknown_websocket_messages_are_ignored(self):
        client = object.__new__(WsClient)

        self.assertIsNone(client.handle_unhandled_message({"type": "future/message"}))


if __name__ == "__main__":
    unittest.main()
