import asyncio
import unittest
from unittest.mock import patch
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.database import Base
from app.core.config import Settings
from app.models.user import User
from app.models.api_key import ApiKey
from app.services.auth_service import seed_default_admin, authenticate_user


class TestFirstUser(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        self.session_maker = async_sessionmaker(
            bind=self.engine, class_=AsyncSession, expire_on_commit=False
        )
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_seed_first_user_from_env(self):
        test_settings = Settings(
            FIRST_SUPERUSER_EMAIL="superadmin@test.local",
            FIRST_SUPERUSER_PASSWORD="SuperSecretPassword123!",
            FIRST_SUPERUSER_NAME="Super Boss",
        )

        with patch("app.services.auth_service.settings", test_settings):
            async with self.session_maker() as session:
                user = await seed_default_admin(session)
                self.assertIsNotNone(user)
                self.assertEqual(user.email, "superadmin@test.local")
                self.assertEqual(user.full_name, "Super Boss")
                self.assertTrue(user.is_admin)
                self.assertTrue(user.is_superadmin)
                self.assertTrue(user.is_active)
                self.assertIsNotNone(user.api_key)

            # Test authentication with seeded password
            async with self.session_maker() as session:
                authenticated = await authenticate_user(
                    session, "superadmin@test.local", "SuperSecretPassword123!"
                )
                self.assertIsNotNone(authenticated)
                self.assertEqual(authenticated.email, "superadmin@test.local")

                # Test wrong password fails
                wrong = await authenticate_user(
                    session, "superadmin@test.local", "WrongPassword"
                )
                self.assertIsNone(wrong)

            # Idempotency test: seeding again returns existing user
            with patch("app.services.auth_service.settings", test_settings):
                async with self.session_maker() as session:
                    same_user = await seed_default_admin(session)
                    self.assertEqual(same_user.id, user.id)

    def test_settings_aliases(self):
        # Test fallback alias FIRST_USER_EMAIL and ADMIN_PASSWORD
        s = Settings(
            FIRST_USER_EMAIL="alias@test.local",
            ADMIN_PASSWORD="aliaspassword",
        )
        self.assertEqual(s.FIRST_SUPERUSER_EMAIL, "alias@test.local")
        self.assertEqual(s.FIRST_SUPERUSER_PASSWORD, "aliaspassword")


if __name__ == "__main__":
    unittest.main()
