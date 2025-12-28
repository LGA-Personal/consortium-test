"""
Tests for node identity and receipt signing.
"""

import pytest
import time

from consortium.identity.keys import KeyManager
from consortium.identity.signing import (
    ReceiptData,
    ReceiptSigner,
    SignedReceipt,
    create_and_sign_receipt,
)


class TestKeyManager:
    """Tests for key generation and management."""

    def test_key_generation(self):
        """Test automatic key generation."""
        km = KeyManager()

        assert km.private_key is not None
        assert km.public_key is not None
        assert len(km.node_id) == 16  # 16 hex chars

    def test_node_id_is_deterministic(self):
        """Same key should produce same node ID."""
        km = KeyManager()
        node_id1 = km.node_id
        node_id2 = km._derive_node_id()
        assert node_id1 == node_id2

    def test_different_keys_different_ids(self):
        """Different keys should produce different node IDs."""
        km1 = KeyManager()
        km2 = KeyManager()
        assert km1.node_id != km2.node_id

    def test_sign_and_verify(self):
        """Test signing and verification."""
        km = KeyManager()
        data = b"test data to sign"

        signature = km.sign(data)
        assert len(signature) == 64  # Ed25519 signature is 64 bytes

        assert km.verify(signature, data) is True
        assert km.verify(signature, b"wrong data") is False

    def test_verify_with_public_key(self):
        """Test verification using raw public key bytes."""
        km = KeyManager()
        data = b"test data"
        signature = km.sign(data)

        # Verify using static method
        assert KeyManager.verify_with_public_key(
            km.public_key_bytes, signature, data
        ) is True

    def test_save_and_load(self, temp_dir):
        """Test saving and loading keys."""
        km1 = KeyManager()
        km1.save(temp_dir)

        # Check files were created
        assert (temp_dir / "private_key.pem").exists()
        assert (temp_dir / "public_key.pem").exists()
        assert (temp_dir / "identity.json").exists()

        # Load and verify
        km2 = KeyManager.load(temp_dir)
        assert km2.node_id == km1.node_id
        assert km2.public_key_bytes == km1.public_key_bytes

        # Verify signature compatibility
        data = b"test data"
        signature = km1.sign(data)
        assert km2.verify(signature, data) is True


class TestReceiptSigning:
    """Tests for receipt signing and verification."""

    @pytest.fixture
    def sample_receipt(self):
        """Create a sample receipt for testing."""
        return ReceiptData(
            session_id="session-123",
            order_id="order-456",
            node_id="node-789",
            token_index=10,
            stage_id=1,
            commitment=b"\x00" * 32,
            input_hash=b"\xff" * 32,
            timestamp_ms=int(time.time() * 1000),
        )

    def test_sign_receipt(self, sample_receipt):
        """Test receipt signing."""
        km = KeyManager()
        signer = ReceiptSigner(km)

        signature = signer.sign_receipt(sample_receipt)
        assert len(signature) == 64

    def test_verify_receipt(self, sample_receipt):
        """Test receipt verification."""
        km = KeyManager()
        signer = ReceiptSigner(km)

        signature = signer.sign_receipt(sample_receipt)
        assert signer.verify_receipt(sample_receipt, signature) is True

    def test_verify_with_different_key(self, sample_receipt):
        """Verification should fail with different key."""
        km1 = KeyManager()
        km2 = KeyManager()

        signer1 = ReceiptSigner(km1)
        signer2 = ReceiptSigner(km2)

        signature = signer1.sign_receipt(sample_receipt)

        # Should fail when verifying with different key
        assert signer2.verify_receipt(sample_receipt, signature) is False

        # Should pass when providing correct public key
        assert signer2.verify_receipt(
            sample_receipt, signature, km1.public_key_bytes
        ) is True

    def test_modified_receipt_fails(self, sample_receipt):
        """Modified receipt should fail verification."""
        km = KeyManager()
        signer = ReceiptSigner(km)

        signature = signer.sign_receipt(sample_receipt)

        # Modify the receipt
        modified = ReceiptData(
            session_id=sample_receipt.session_id,
            order_id=sample_receipt.order_id,
            node_id=sample_receipt.node_id,
            token_index=sample_receipt.token_index + 1,  # Changed!
            stage_id=sample_receipt.stage_id,
            commitment=sample_receipt.commitment,
            input_hash=sample_receipt.input_hash,
            timestamp_ms=sample_receipt.timestamp_ms,
        )

        assert signer.verify_receipt(modified, signature) is False


class TestSignedReceipt:
    """Tests for the SignedReceipt dataclass."""

    def test_create_and_sign_receipt(self):
        """Test the convenience function."""
        km = KeyManager()

        signed = create_and_sign_receipt(
            key_manager=km,
            session_id="session-123",
            order_id="order-456",
            token_index=10,
            stage_id=1,
            commitment=b"\x00" * 32,
            input_hash=b"\xff" * 32,
            timestamp_ms=12345678,
        )

        assert signed.data.session_id == "session-123"
        assert signed.data.node_id == km.node_id
        assert len(signed.signature) == 64
        assert signed.signer_public_key == km.public_key_bytes
