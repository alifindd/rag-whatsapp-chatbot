from fastapi import APIRouter
from fastapi import Request
from app.schemas.chat import ChatRequest
from app.services.rag_service import chat_with_rag
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse

router = APIRouter()

@router.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    incoming_msg = str(form.get("Body", ""))
    sender = str(form.get("From"))

    answer = chat_with_rag(incoming_msg, sender)

    resp = MessagingResponse()
    resp.message(answer)
    return PlainTextResponse(str(resp), media_type="text/xml")