import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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


class _Response:
    def __init__(self, *, status=200, body=b'body', error=None):
        self.status = status
        self.reason = 'OK'
        self.headers = {}
        self._body = body
        self._error = error

    async def read(self):
        if self._error is not None:
            raise self._error
        return self._body


class TestRestTracing(unittest.IsolatedAsyncioTestCase):
    async def _make_client(self, trace_dir):
        with patch.dict(
            os.environ, {LATENCY_TRACE_DIR_ENV: trace_dir}, clear=False
        ):
            return RESTClientObject(_Configuration())

    async def test_blank_trace_dir_installs_no_hooks(self):
        with patch('lighter.rest.aiohttp.TraceConfig') as trace_config:
            client = await self._make_client('   ')
        try:
            trace_config.assert_not_called()
            self.assertEqual(client.pool_manager.trace_configs, [])
        finally:
            await client.close()

    async def test_opt_in_hooks_emit_transport_fields_and_order(self):
        client = await self._make_client('trace-enabled')
        events = []

        def observer(phase, fields):
            events.append((phase, dict(fields)))

        try:
            trace_config = client.pool_manager.trace_configs[0]
            expected_hooks = {
                'on_request_start': RESTClientObject._trace_request_start,
                'on_connection_queued_start': (
                    RESTClientObject._trace_connection_queued_start
                ),
                'on_connection_queued_end': (
                    RESTClientObject._trace_connection_queued_end
                ),
                'on_dns_resolvehost_start': (
                    RESTClientObject._trace_dns_resolve_start
                ),
                'on_dns_resolvehost_end': RESTClientObject._trace_dns_resolve_end,
                'on_dns_cache_hit': RESTClientObject._trace_dns_cache_hit,
                'on_dns_cache_miss': RESTClientObject._trace_dns_cache_miss,
                'on_connection_create_start': (
                    RESTClientObject._trace_connection_create_start
                ),
                'on_connection_create_end': (
                    RESTClientObject._trace_connection_create_end
                ),
                'on_connection_reuseconn': (
                    RESTClientObject._trace_connection_reuse
                ),
                'on_request_end': RESTClientObject._trace_request_end,
                'on_request_exception': RESTClientObject._trace_request_exception,
            }
            for signal_name, callback in expected_hooks.items():
                self.assertEqual(list(getattr(trace_config, signal_name)), [callback])

            request_params = SimpleNamespace(
                url=SimpleNamespace(scheme='HTTPS', raw_host='api.example.test')
            )
            context = SimpleNamespace()
            response = _Response(status=206, body=b'abc')
            with client.trace_request(observer):
                await client._trace_request_start(
                    client.pool_manager, context, request_params
                )
                await client._trace_connection_queued_start(
                    client.pool_manager, context, SimpleNamespace()
                )
                await client._trace_connection_queued_end(
                    client.pool_manager, context, SimpleNamespace()
                )
                await client._trace_connection_create_start(
                    client.pool_manager, context, SimpleNamespace()
                )
                await client._trace_dns_cache_miss(
                    client.pool_manager, context, SimpleNamespace()
                )
                await client._trace_dns_resolve_start(
                    client.pool_manager, context, SimpleNamespace()
                )
                await client._trace_dns_resolve_end(
                    client.pool_manager, context, SimpleNamespace()
                )
                await client._trace_connection_create_end(
                    client.pool_manager, context, SimpleNamespace()
                )
                await client._trace_request_end(
                    client.pool_manager,
                    context,
                    SimpleNamespace(response=response),
                )
            self.assertEqual(await RESTResponse(response).read(), b'abc')

            cache_context = SimpleNamespace()
            cache_response = _Response()
            with client.trace_request(observer):
                await client._trace_request_start(
                    client.pool_manager, cache_context, request_params
                )
                await client._trace_connection_create_start(
                    client.pool_manager, cache_context, SimpleNamespace()
                )
                await client._trace_dns_cache_hit(
                    client.pool_manager, cache_context, SimpleNamespace()
                )
                await client._trace_connection_create_end(
                    client.pool_manager, cache_context, SimpleNamespace()
                )
                await client._trace_request_end(
                    client.pool_manager,
                    cache_context,
                    SimpleNamespace(response=cache_response),
                )
            await RESTResponse(cache_response).read()

            reuse_context = SimpleNamespace()
            reuse_response = _Response()
            with client.trace_request(observer):
                await client._trace_request_start(
                    client.pool_manager, reuse_context, request_params
                )
                await client._trace_connection_reuse(
                    client.pool_manager, reuse_context, SimpleNamespace()
                )
                await client._trace_request_end(
                    client.pool_manager,
                    reuse_context,
                    SimpleNamespace(response=reuse_response),
                )
            await RESTResponse(reuse_response).read()

            self.assertEqual(
                [phase for phase, _ in events],
                [
                    'request_start',
                    'connection_queued_start',
                    'connection_queued_end',
                    'connection_create_start',
                    'dns_cache_miss',
                    'dns_resolve_start',
                    'dns_resolve_end',
                    'connection_create_end',
                    'response_headers',
                    'response_body_end',
                    'request_start',
                    'connection_create_start',
                    'dns_cache_hit',
                    'connection_create_end',
                    'response_headers',
                    'response_body_end',
                    'request_start',
                    'connection_reuse',
                    'response_headers',
                    'response_body_end',
                ],
            )

            connector = client.pool_manager.connector
            request_fields = events[0][1]
            self.assertEqual(
                request_fields,
                {
                    'transport_trace_version': 1,
                    'scheme': 'https',
                    'connection_state': 'unknown',
                    'dns_resolution_expected': True,
                    'keepalive_timeout': connector._keepalive_timeout,
                    'dns_cache_enabled': connector.use_dns_cache,
                    'connection_limit': connector.limit,
                    'connection_limit_per_host': connector.limit_per_host,
                    'force_close': connector.force_close,
                    'dns_cache_ttl': connector._cached_hosts._ttl,
                },
            )
            self.assertEqual(
                events[3][1],
                {
                    'connection_state': 'new',
                    'scheme': 'https',
                    'connection_scope': 'dns_tcp_tls_combined',
                },
            )
            self.assertEqual(
                events[7][1],
                {
                    'connection_state': 'new',
                    'scheme': 'https',
                    'connection_scope': 'dns_tcp_tls_combined',
                    'tls_separately_observable': False,
                },
            )
            self.assertEqual(
                events[8][1],
                {'connection_state': 'new', 'http_status': 206},
            )
            self.assertEqual(
                events[9][1],
                {'connection_state': 'new', 'response_bytes': 3},
            )
            self.assertEqual(
                events[17][1],
                {'connection_state': 'reused', 'scheme': 'https'},
            )
        finally:
            await client.close()

    async def test_error_events_retain_connection_state(self):
        client = await self._make_client('trace-enabled')
        events = []

        def observer(phase, fields):
            events.append((phase, dict(fields)))

        request_params = SimpleNamespace(
            url=SimpleNamespace(scheme='https', raw_host='api.example.test')
        )
        try:
            request_context = SimpleNamespace()
            with client.trace_request(observer):
                await client._trace_request_start(
                    client.pool_manager, request_context, request_params
                )
                await client._trace_connection_create_start(
                    client.pool_manager, request_context, SimpleNamespace()
                )
                await client._trace_request_exception(
                    client.pool_manager,
                    request_context,
                    SimpleNamespace(exception=TimeoutError()),
                )

            body_context = SimpleNamespace()
            response = _Response(error=OSError('read failed'))
            with client.trace_request(observer):
                await client._trace_request_start(
                    client.pool_manager, body_context, request_params
                )
                await client._trace_connection_reuse(
                    client.pool_manager, body_context, SimpleNamespace()
                )
                await client._trace_request_end(
                    client.pool_manager,
                    body_context,
                    SimpleNamespace(response=response),
                )
            with self.assertRaisesRegex(OSError, 'read failed'):
                await RESTResponse(response).read()

            self.assertEqual(
                [phase for phase, _ in events],
                [
                    'request_start',
                    'connection_create_start',
                    'request_error',
                    'request_start',
                    'connection_reuse',
                    'response_headers',
                    'response_body_error',
                ],
            )
            self.assertEqual(
                events[2][1],
                {'connection_state': 'new', 'error_type': 'TimeoutError'},
            )
            self.assertEqual(
                events[6][1],
                {'connection_state': 'reused', 'error_type': 'OSError'},
            )
        finally:
            await client.close()

    async def test_request_start_classifies_ip_literal_without_exposing_it(self):
        client = await self._make_client('trace-enabled')
        events = []

        try:
            for host in ('127.0.0.1', '::1'):
                context = SimpleNamespace()
                with client.trace_request(
                    lambda phase, fields: events.append((phase, dict(fields)))
                ):
                    await client._trace_request_start(
                        client.pool_manager,
                        context,
                        SimpleNamespace(
                            url=SimpleNamespace(scheme='https', raw_host=host)
                        ),
                    )
                self.assertFalse(context.btcfine_dns_resolution_expected)

            request_starts = [
                fields for phase, fields in events if phase == 'request_start'
            ]
            self.assertEqual(
                [fields['dns_resolution_expected'] for fields in request_starts],
                [False, False],
            )
            for fields in request_starts:
                self.assertNotIn('host', fields)
                self.assertNotIn('raw_host', fields)
        finally:
            await client.close()


if __name__ == '__main__':
    unittest.main()
