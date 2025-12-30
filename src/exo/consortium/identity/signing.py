"""
Receipt Signing and Verification

This module provides functions for signing and verifying receipts
in the Consortium protocol.
"""

import struct
from dataclasses import dataclass
from typing import Optional

from .keys import KeyManager


@dataclass
class ReceiptData:
    """Data fields that make up a receipt."""

    session_id: str
    order_id: str
    node_id: str
    token_index: int
    stage_id: int
    commitment: bytes  # 32-byte SHA-256 hash
    input_hash: bytes  # 32-byte SHA-256 hash of input
    timestamp_ms: int


class ReceiptSigner:
    """Signs and verifies receipts using Ed25519."""

    def __init__(self, key_manager: KeyManager):
        """
        Initialize signer with key manager.

        Args:
            key_manager: KeyManager instance with private key for signing
        """
        self.key_manager = key_manager

    def _canonical_bytes(self, receipt: ReceiptData) -> bytes:
        """
        Create canonical byte representation for signing.

        The order and format of fields is fixed to ensure
        deterministic signatures.
        """
        return (
            receipt.session_id.encode("utf-8")
            + b"\x00"  # Null separator
            + receipt.order_id.encode("utf-8")
            + b"\x00"
            + receipt.node_id.encode("utf-8")
            + b"\x00"
            + struct.pack("<II", receipt.token_index, receipt.stage_id)
            + receipt.commitment
            + receipt.input_hash
            + struct.pack("<Q", receipt.timestamp_ms)
        )

    def sign_receipt(self, receipt: ReceiptData) -> bytes:
        """
        Sign a receipt.

        Args:
            receipt: Receipt data to sign

        Returns:
            64-byte Ed25519 signature
        """
        canonical = self._canonical_bytes(receipt)
        return self.key_manager.sign(canonical)

    def verify_receipt(
        self,
        receipt: ReceiptData,
        signature: bytes,
        public_key_bytes: Optional[bytes] = None,
    ) -> bool:
        """
        Verify a receipt signature.

        Args:
            receipt: Receipt data that was signed
            signature: 64-byte Ed25519 signature
            public_key_bytes: Public key of signer (uses own key if None)

        Returns:
            True if signature is valid
        """
        canonical = self._canonical_bytes(receipt)

        if public_key_bytes is None:
            return self.key_manager.verify(signature, canonical)
        else:
            return KeyManager.verify_with_public_key(
                public_key_bytes, signature, canonical
            )


@dataclass
class SignedReceipt:
    """A receipt with its signature."""

    data: ReceiptData
    signature: bytes
    signer_public_key: bytes  # 32-byte raw public key

    def verify(self) -> bool:
        """Verify this receipt's signature."""
        return KeyManager.verify_with_public_key(
            self.signer_public_key,
            self.signature,
            ReceiptSigner._canonical_bytes_static(self.data),
        )

    @staticmethod
    def _canonical_bytes_static(receipt: ReceiptData) -> bytes:
        """Static version for use in dataclass method."""
        return (
            receipt.session_id.encode("utf-8")
            + b"\x00"
            + receipt.order_id.encode("utf-8")
            + b"\x00"
            + receipt.node_id.encode("utf-8")
            + b"\x00"
            + struct.pack("<II", receipt.token_index, receipt.stage_id)
            + receipt.commitment
            + receipt.input_hash
            + struct.pack("<Q", receipt.timestamp_ms)
        )


def create_and_sign_receipt(
    key_manager: KeyManager,
    session_id: str,
    order_id: str,
    token_index: int,
    stage_id: int,
    commitment: bytes,
    input_hash: bytes,
    timestamp_ms: int,
) -> SignedReceipt:
    """
    Create and sign a receipt in one step.

    Args:
        key_manager: Key manager with signing key
        session_id: Session identifier
        order_id: Work order identifier
        token_index: Token position in sequence
        stage_id: Pipeline stage (0, 1, or 2)
        commitment: Output commitment hash
        input_hash: Hash of input activation
        timestamp_ms: Unix timestamp in milliseconds

    Returns:
        SignedReceipt with data and signature
    """
    signer = ReceiptSigner(key_manager)

    receipt_data = ReceiptData(
        session_id=session_id,
        order_id=order_id,
        node_id=key_manager.node_id,
        token_index=token_index,
        stage_id=stage_id,
        commitment=commitment,
        input_hash=input_hash,
        timestamp_ms=timestamp_ms,
    )

    signature = signer.sign_receipt(receipt_data)

    return SignedReceipt(
        data=receipt_data,
        signature=signature,
        signer_public_key=key_manager.public_key_bytes,
    )
