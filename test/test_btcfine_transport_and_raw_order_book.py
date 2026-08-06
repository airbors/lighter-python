import asyncio
import os
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, AsyncMock, patch

from pydantic import ValidationError

from lighter.api.order_api import OrderApi
from lighter.api_client import ApiClient
from lighter.exceptions import ApiException, BadRequestException
from lighter.models.order_book_orders import OrderBookOrders
from lighter.rest import LATENCY_TRACE_DIR_ENV, RESTClientObject, RESTResponse


class _Configuration:
    connection_pool_maxsize = 3
    ssl_ca_cert = None
    cert_file = None
    key_file = None
    verify_ssl = True
    proxy = None
    proxy_headers = None
    retries = None


class _RawResponse:
    def __init__(
        self,
        *,
        status=200,
        reason="OK",
        body=b"body",
        content_type="application/json",
    ):
        self.status = status
        self.reason = reason
        self.headers = {"content-type": content_type}
        self.body = body
        self.read_count = 0

    async def read(self):
        self.read_count += 1
        return self.body


class _ReadResult:
    def __init__(self, body):
        self.status = 200
        self.reason = "OK"
        self.data = body
        self.read_count = 0

    async def read(self):
        self.read_count += 1
        return self.data

    def getheaders(self):
        return {}


def _api_client(response_data):
    client = object.__new__(ApiClient)
    client.configuration = SimpleNamespace(
        host="https://api.example.test",
        ignore_operation_servers=True,
        safe_chars_for_path_param="",
    )
    client.default_headers = {}
    client.cookie = None
    client.call_api = AsyncMock(return_value=response_data)
    return client


class TestBtcfineConnectorSettings(unittest.IsolatedAsyncioTestCase):
    async def test_exact_connector_values_and_single_session_topology(self):
        connector = object()
        session = SimpleNamespace(close=AsyncMock())

        with (
            patch.dict(
                os.environ, {LATENCY_TRACE_DIR_ENV: ""}, clear=False
            ),
            patch(
                "lighter.rest.aiohttp.TCPConnector", return_value=connector
            ) as connector_factory,
            patch(
                "lighter.rest.aiohttp.ClientSession", return_value=session
            ) as session_factory,
        ):
            client = RESTClientObject(_Configuration())

        connector_factory.assert_called_once_with(
            limit=3,
            ssl=ANY,
            keepalive_timeout=45.0,
            ttl_dns_cache=45,
        )
        session_factory.assert_called_once_with(
            connector=connector,
            trust_env=True,
            trace_configs=[],
        )
        self.assertIs(client.pool_manager, session)
        self.assertIsNone(client.retry_client)
        await client.close()
        session.close.assert_awaited_once()

    async def test_cancelled_request_keeps_same_session_reusable(self):
        with patch.dict(
            os.environ, {LATENCY_TRACE_DIR_ENV: ""}, clear=False
        ):
            client = RESTClientObject(_Configuration())
        session = client.pool_manager
        connector = session.connector
        self.assertEqual(connector._keepalive_timeout, 45.0)
        self.assertEqual(connector._cached_hosts._ttl, 45)
        started = asyncio.Event()
        calls = []

        async def request(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                started.set()
                await asyncio.Future()
            return _RawResponse(body=b"later request")

        try:
            with patch.object(session, "request", new=request):
                first = asyncio.create_task(
                    client.request(
                        "GET",
                        "https://api.example.test/first",
                        _request_timeout=2.5,
                    )
                )
                await started.wait()
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first

                self.assertFalse(session.closed)
                self.assertIs(client.pool_manager, session)
                self.assertIs(session.connector, connector)

                response = await client.request(
                    "GET",
                    "https://api.example.test/second",
                    _request_timeout=2.5,
                )
                self.assertEqual(await response.read(), b"later request")

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["timeout"], 2.5)
            self.assertEqual(calls[1]["timeout"], 2.5)
        finally:
            await client.close()


class TestBtcfineRawOrderBook(unittest.IsolatedAsyncioTestCase):
    async def test_raw_seam_uses_generated_request_and_reads_once(self):
        body = (
            b'{"code":200,"total_asks":1,"asks":[],'
            b'"total_bids":1,"bids":[]}'
        )
        raw_response = _RawResponse(body=body)
        response_data = RESTResponse(raw_response)
        trace_events = []
        response_data._trace_observer = lambda phase, fields: trace_events.append(
            (phase, dict(fields))
        )
        response_data._connection_state = "reused"
        api_client = _api_client(response_data)
        order_api = OrderApi(api_client)

        result = await order_api.btcfine_order_book_orders_raw(
            market_id=7,
            limit=250,
            _request_timeout=2.5,
        )

        self.assertIs(result, body)
        self.assertEqual(raw_response.read_count, 1)
        api_client.call_api.assert_awaited_once_with(
            "GET",
            (
                "https://api.example.test/api/v1/orderBookOrders"
                "?market_id=7&limit=250"
            ),
            {"Accept": "application/json"},
            None,
            [],
            _request_timeout=2.5,
        )
        self.assertEqual(
            trace_events,
            [
                (
                    "response_body_end",
                    {"connection_state": "reused", "response_bytes": len(body)},
                )
            ],
        )

    async def test_raw_seam_retains_generated_http_error_behavior(self):
        body = b'{"code":400,"message":"bad book request"}'
        raw_response = _RawResponse(
            status=400,
            reason="Bad Request",
            body=body,
        )
        api_client = _api_client(RESTResponse(raw_response))
        order_api = OrderApi(api_client)

        with self.assertRaises(BadRequestException) as raised:
            await order_api.btcfine_order_book_orders_raw(
                market_id=7,
                limit=250,
                _request_timeout=2.5,
            )

        self.assertEqual(raw_response.read_count, 1)
        api_client.call_api.assert_awaited_once()
        self.assertEqual(raised.exception.status, 400)
        self.assertEqual(raised.exception.body, body.decode("utf-8"))
        self.assertEqual(raised.exception.data.code, 400)
        self.assertEqual(raised.exception.data.message, "bad book request")

    async def test_raw_seam_rejects_unexpected_success_status(self):
        body = b'{"code":201}'
        raw_response = _RawResponse(status=201, body=body)
        api_client = _api_client(RESTResponse(raw_response))
        order_api = OrderApi(api_client)

        with self.assertRaises(ApiException) as raised:
            await order_api.btcfine_order_book_orders_raw(
                market_id=7,
                limit=250,
                _request_timeout=2.5,
            )

        self.assertEqual(raw_response.read_count, 1)
        api_client.call_api.assert_awaited_once()
        self.assertEqual(raised.exception.status, 201)
        self.assertEqual(raised.exception.body, body.decode("utf-8"))
        self.assertIn("Unexpected success status", raised.exception.reason)

    async def test_raw_seam_requires_exact_bytes_success_body(self):
        for body in (bytearray(b"book"), None):
            with self.subTest(body_type=type(body).__name__):
                response_data = _ReadResult(body)
                api_client = _api_client(response_data)
                order_api = OrderApi(api_client)

                with self.assertRaises(ApiException) as raised:
                    await order_api.btcfine_order_book_orders_raw(
                        market_id=7,
                        limit=250,
                        _request_timeout=2.5,
                    )

                self.assertEqual(response_data.read_count, 1)
                api_client.call_api.assert_awaited_once()
                self.assertEqual(raised.exception.status, 0)
                self.assertIn("must be exact bytes", raised.exception.reason)

    async def test_raw_seam_rejects_invalid_inputs_before_request(self):
        cases = (
            {"market_id": 7, "limit": 251, "_request_timeout": 2.5},
            {"market_id": 7, "limit": 0, "_request_timeout": 2.5},
            {"market_id": 7, "limit": True, "_request_timeout": 2.5},
            {"market_id": "7", "limit": 250, "_request_timeout": 2.5},
            {"market_id": 7, "limit": 250, "_request_timeout": 0.0},
        )

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                api_client = _api_client(RESTResponse(_RawResponse()))
                order_api = OrderApi(api_client)
                with self.assertRaises(ValidationError):
                    await order_api.btcfine_order_book_orders_raw(**kwargs)
                api_client.call_api.assert_not_awaited()

    async def test_generated_order_book_method_still_returns_model(self):
        body = (
            b'{"code":200,"total_asks":0,"asks":[],'
            b'"total_bids":0,"bids":[]}'
        )
        api_client = _api_client(RESTResponse(_RawResponse(body=body)))
        order_api = OrderApi(api_client)

        result = await order_api.order_book_orders(
            market_id=7,
            limit=250,
            _request_timeout=2.5,
        )

        self.assertIsInstance(result, OrderBookOrders)
        api_client.call_api.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
