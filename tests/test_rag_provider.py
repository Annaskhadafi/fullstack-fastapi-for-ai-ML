import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from app.services import rag_service, rag_vectorless_service


class SharedProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_pgvector_uses_vectorless_chat_settings(self):
        previous = rag_service._provider.copy()
        try:
            rag_vectorless_service.update_provider("test-key", "custom-model", "https://example.test/v1/")
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="Jawaban uji"))])))))
            with patch.object(rag_service, "AsyncOpenAI", return_value=client) as factory, patch.object(
                rag_service, "search_similar_documents", new=AsyncMock(return_value=[])
            ):
                result = await rag_service.answer_rag_query(None, "Pertanyaan")
            factory.assert_called_once_with(api_key="test-key", base_url="https://example.test/v1")
            self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], "custom-model")
            self.assertEqual(result.answer, "Jawaban uji")
            self.assertNotIn("api_key", rag_service.provider_settings())
        finally:
            rag_service._provider.update(previous)
