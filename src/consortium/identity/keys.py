"""
Node Identity and Key Management

This module provides Ed25519 key generation, storage, and loading
for node identity in the Consortium network.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


class KeyManager:
    """Manages Ed25519 keys for node identity."""

    def __init__(
        self,
        private_key: Optional[Ed25519PrivateKey] = None,
    ):
        """
        Initialize key manager.

        Args:
            private_key: Existing private key, or None to generate new
        """
        self._private_key = private_key or Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()
        self._node_id = self._derive_node_id()

    @property
    def private_key(self) -> Ed25519PrivateKey:
        """Get the private key."""
        return self._private_key

    @property
    def public_key(self) -> Ed25519PublicKey:
        """Get the public key."""
        return self._public_key

    @property
    def node_id(self) -> str:
        """
        Get the node ID derived from the public key.

        The node ID is the first 16 hex characters of the SHA-256 hash
        of the raw public key bytes.
        """
        return self._node_id

    def _derive_node_id(self) -> str:
        """Derive node ID from public key."""
        pub_bytes = self.public_key_bytes
        return hashlib.sha256(pub_bytes).hexdigest()[:16]

    @property
    def public_key_bytes(self) -> bytes:
        """Get raw public key bytes."""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @property
    def private_key_bytes(self) -> bytes:
        """Get raw private key bytes (for secure storage)."""
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def sign(self, data: bytes) -> bytes:
        """
        Sign data with the private key.

        Args:
            data: Bytes to sign

        Returns:
            64-byte Ed25519 signature
        """
        return self._private_key.sign(data)

    def verify(self, signature: bytes, data: bytes) -> bool:
        """
        Verify a signature with this node's public key.

        Args:
            signature: 64-byte Ed25519 signature
            data: Original data that was signed

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            self._public_key.verify(signature, data)
            return True
        except Exception:
            return False

    def save(self, path: Path) -> None:
        """
        Save keys to a directory.

        Creates:
        - private_key.pem: PEM-encoded private key
        - public_key.pem: PEM-encoded public key
        - identity.json: Node ID and public key hex

        Args:
            path: Directory to save keys to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save private key
        private_pem = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        (path / "private_key.pem").write_bytes(private_pem)

        # Save public key
        public_pem = self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        (path / "public_key.pem").write_bytes(public_pem)

        # Save identity info
        identity = {
            "node_id": self.node_id,
            "public_key_hex": self.public_key_bytes.hex(),
        }
        (path / "identity.json").write_text(json.dumps(identity, indent=2))

    @classmethod
    def load(cls, path: Path) -> "KeyManager":
        """
        Load keys from a directory.

        Args:
            path: Directory containing private_key.pem

        Returns:
            KeyManager instance
        """
        path = Path(path)
        private_pem = (path / "private_key.pem").read_bytes()
        private_key = serialization.load_pem_private_key(
            private_pem,
            password=None,
        )

        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError("Expected Ed25519 private key")

        return cls(private_key=private_key)

    @classmethod
    def load_public_key(cls, public_key_bytes: bytes) -> Ed25519PublicKey:
        """
        Load a public key from raw bytes.

        Args:
            public_key_bytes: 32-byte raw public key

        Returns:
            Ed25519PublicKey instance
        """
        return Ed25519PublicKey.from_public_bytes(public_key_bytes)

    @classmethod
    def verify_with_public_key(
        cls,
        public_key_bytes: bytes,
        signature: bytes,
        data: bytes,
    ) -> bool:
        """
        Verify a signature with a public key.

        Args:
            public_key_bytes: 32-byte raw public key
            signature: 64-byte Ed25519 signature
            data: Original data that was signed

        Returns:
            True if signature is valid
        """
        try:
            public_key = cls.load_public_key(public_key_bytes)
            public_key.verify(signature, data)
            return True
        except Exception:
            return False
