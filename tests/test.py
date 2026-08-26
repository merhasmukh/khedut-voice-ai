from ai_services.whatsapp_client import send_whatsapp_message

response = send_whatsapp_message(
    to=["919724455986"],
    body_params=["હસમુખભાઈ"],
    template_name="natural_farming_ai_bot_response",
    language="gu",
)

print(response)