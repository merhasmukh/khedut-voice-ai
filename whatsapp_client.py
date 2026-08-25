"""
WhatsApp Notification Client for Khedut Voice AI
==================================================
Converts WhatsApp template messaging service to Python with full async + sync support.
All credentials (API key, service URL) are securely loaded from .env.

Supported Features:
  - Send template WhatsApp messages with body_params
  - Optional button_params for templates with dynamic action buttons
  - Supports single recipient (str) or multiple recipients (list of str)
  - Both Async (FastAPI / Gemini Live background tasks) and Sync (scripts / CLI)
  - Comprehensive error handling and response status validation

Environment Variables (.env):
  WHATSAPP_API_KEY=your_api_key_here
  WHATSAPP_SERVICE_URL=https://web.gujaratvidyapith.org/services/whatsapp_service/api_whatsapp.php
  WHATSAPP_DEFAULT_TEMPLATE=logincred
  WHATSAPP_DEFAULT_LANGUAGE=en

Usage Example (Sync):
  from whatsapp_client import send_whatsapp_message

  res = send_whatsapp_message(
      to=["919876543210"],
      body_params=["હસમુખભાઈ", "જીવામૃત રેસિપી લિંક"],
      template_name="logincred",
      language="en"
  )
  print(res)

Usage Example (Async in FastAPI / Voice AI):
  from whatsapp_client import send_whatsapp_message_async

  res = await send_whatsapp_message_async(
      to="919876543210",
      body_params=["હસમુખભાઈ", "પ્રાકૃતિક ખેતી ગાઈડ"],
  )
"""

import os
from typing import Any, Dict, List, Optional, Union
import httpx
from dotenv import load_dotenv

load_dotenv()

# ─── Default Configuration from Environment ──────────────────────────────────
DEFAULT_WHATSAPP_API_KEY = os.environ.get("WHATSAPP_API_KEY", "").strip()
DEFAULT_WHATSAPP_SERVICE_URL = os.environ.get(
    "WHATSAPP_SERVICE_URL",
    "https://web.gujaratvidyapith.org/services/whatsapp_service/api_whatsapp.php"
).strip()
DEFAULT_TEMPLATE_NAME = os.environ.get("WHATSAPP_DEFAULT_TEMPLATE", "logincred").strip()
DEFAULT_LANGUAGE = os.environ.get("WHATSAPP_DEFAULT_LANGUAGE", "en").strip()


class WhatsAppClient:
    """
    Client for interacting with the Gujarat Vidyapith WhatsApp Gateway Service.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        service_url: Optional[str] = None,
        timeout: float = 15.0,
    ):
        self.api_key = (api_key or DEFAULT_WHATSAPP_API_KEY or "").strip()
        self.service_url = (service_url or DEFAULT_WHATSAPP_SERVICE_URL).strip()
        self.timeout = timeout

    def _build_payload(
        self,
        to: Union[str, List[str]],
        body_params: Optional[List[str]] = None,
        button_params: Optional[List[str]] = None,
        template_name: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Constructs and validates the JSON payload expected by the WhatsApp API."""
        if not self.api_key:
            raise ValueError(
                "WHATSAPP_API_KEY is not set. "
                "Please configure WHATSAPP_API_KEY in your .env file."
            )

        # Normalize recipients to a list of non-empty strings
        if isinstance(to, str):
            recipients = [to.strip()] if to.strip() else []
        elif isinstance(to, list):
            recipients = [str(num).strip() for num in to if str(num).strip()]
        else:
            recipients = []

        if not recipients:
            raise ValueError("Recipient 'to' phone number(s) must be provided.")

        payload: Dict[str, Any] = {
            "api_key": self.api_key,
            "to": recipients,
            "template_name": template_name or DEFAULT_TEMPLATE_NAME,
            "language": language or DEFAULT_LANGUAGE,
            "type": "template",
            "body_params": body_params if body_params is not None else [],
        }

        # Only attach button_params if explicitly provided
        if button_params:
            payload["button_params"] = button_params

        return payload

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "API-KEY": self.api_key,
        }

    # ── Synchronous Send ──────────────────────────────────────────────────────
    def send_template(
        self,
        to: Union[str, List[str]],
        body_params: Optional[List[str]] = None,
        button_params: Optional[List[str]] = None,
        template_name: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sends a WhatsApp template message synchronously.

        Returns:
            dict containing response from WhatsApp gateway or error description.
        """
        payload = self._build_payload(
            to=to,
            body_params=body_params,
            button_params=button_params,
            template_name=template_name,
            language=language,
        )
        headers = self._get_headers()

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.service_url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                try:
                    return response.json()
                except Exception:
                    return {"status": "success", "raw_response": response.text}
        except httpx.HTTPStatusError as exc:
            return {
                "status": "error",
                "error": f"HTTP {exc.response.status_code}: {exc.response.text}",
                "status_code": exc.response.status_code,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ── Asynchronous Send ─────────────────────────────────────────────────────
    async def send_template_async(
        self,
        to: Union[str, List[str]],
        body_params: Optional[List[str]] = None,
        button_params: Optional[List[str]] = None,
        template_name: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sends a WhatsApp template message asynchronously.
        Ideal for use inside FastAPI endpoints or background async workers.

        Returns:
            dict containing response from WhatsApp gateway or error description.
        """
        payload = self._build_payload(
            to=to,
            body_params=body_params,
            button_params=button_params,
            template_name=template_name,
            language=language,
        )
        headers = self._get_headers()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.service_url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                try:
                    return response.json()
                except Exception:
                    return {"status": "success", "raw_response": response.text}
        except httpx.HTTPStatusError as exc:
            return {
                "status": "error",
                "error": f"HTTP {exc.response.status_code}: {exc.response.text}",
                "status_code": exc.response.status_code,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}


# ── Global Singleton & Convenience Functions ─────────────────────────────────

_default_client: Optional[WhatsAppClient] = None


def get_whatsapp_client() -> WhatsAppClient:
    """Returns or creates the default singleton WhatsAppClient instance."""
    global _default_client
    if _default_client is None:
        _default_client = WhatsAppClient()
    return _default_client


def send_whatsapp_message(
    to: Union[str, List[str]],
    body_params: Optional[List[str]] = None,
    button_params: Optional[List[str]] = None,
    template_name: Optional[str] = None,
    language: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to send a WhatsApp template message synchronously.
    """
    client = WhatsAppClient(api_key=api_key) if api_key else get_whatsapp_client()
    return client.send_template(
        to=to,
        body_params=body_params,
        button_params=button_params,
        template_name=template_name,
        language=language,
    )


async def send_whatsapp_message_async(
    to: Union[str, List[str]],
    body_params: Optional[List[str]] = None,
    button_params: Optional[List[str]] = None,
    template_name: Optional[str] = None,
    language: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to send a WhatsApp template message asynchronously.
    """
    client = WhatsAppClient(api_key=api_key) if api_key else get_whatsapp_client()
    return await client.send_template_async(
        to=to,
        body_params=body_params,
        button_params=button_params,
        template_name=template_name,
        language=language,
    )


# ── CLI / Direct Execution Test ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="WhatsApp Service CLI Tester")
    parser.add_argument("--to", nargs="+", help="Recipient phone number(s) (e.g. 919876543210)")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE_NAME, help="Template name")
    parser.add_argument("--body-params", nargs="*", default=[], help="Body parameter list")
    parser.add_argument("--button-params", nargs="*", default=None, help="Button parameter list (optional)")
    parser.add_argument("--api-key", default=None, help="WhatsApp API Key (overrides .env)")
    args = parser.parse_args()

    print("══════════════════════════════════════════════")
    print(" 📲 WhatsApp Notification Service (Python)")
    print("══════════════════════════════════════════════")
    
    if not args.to:
        print("\nℹ️ Usage: python whatsapp_client.py --to 919876543210 --body-params 'Param1' 'Param2'")
        print(f"Current Service URL: {DEFAULT_WHATSAPP_SERVICE_URL}")
        print(f"API Key configured in .env: {'✅ Yes' if DEFAULT_WHATSAPP_API_KEY else '❌ No (set WHATSAPP_API_KEY in .env)'}")
    else:
        print(f"Sending to: {args.to}")
        print(f"Template:   {args.template}")
        print(f"Body:       {args.body_params}")
        res = send_whatsapp_message(
            to=args.to,
            body_params=args.body_params,
            button_params=args.button_params,
            template_name=args.template,
            api_key=args.api_key,
        )
        print("\nResponse:")
        print(json.dumps(res, indent=2, ensure_ascii=False))
