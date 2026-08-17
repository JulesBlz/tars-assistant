"""
Client Gmail pour TARS. Lecture seule.
OAuth au premier lancement, token stocké localement pour les suivants.
"""
import os
import base64
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/calendar.readonly',
]
CREDENTIALS_PATH = os.path.expanduser("~/tars/credentials/gmail_credentials.json")
TOKEN_PATH = os.path.expanduser("~/tars/credentials/google_token.json")


def get_google_credentials():
    """Retourne des credentials Google valides (Gmail + Calendar). Lance OAuth si nécessaire."""
    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())

    return creds


def get_gmail_service():
    """Retourne un service Gmail authentifié."""
    creds = get_google_credentials()
    return build('gmail', 'v1', credentials=creds)


def extract_body(payload):
    """Extrait le corps texte d'un message Gmail."""
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain' and 'data' in part.get('body', {}):
                return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
            if 'parts' in part:
                result = extract_body(part)
                if result:
                    return result
    if payload.get('mimeType') == 'text/plain' and 'data' in payload.get('body', {}):
        return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
    return ""


def search_emails(query="", max_results=10, days_back=30):
    """
    Cherche des emails via l'API Gmail.
    
    query: syntaxe Gmail (ex: "from:recruiter", "subject:entretien", "is:unread")
    max_results: nombre max de résultats
    days_back: fenêtre temporelle (jours)
    
    Retourne une liste de dicts avec: from, subject, date, snippet, body.
    """
    service = get_gmail_service()

    date_filter = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
    full_query = f"after:{date_filter}"
    if query:
        full_query += f" {query}"

    print(f"DEBUG GMAIL: recherche '{full_query}' (max {max_results})", flush=True)

    results = service.users().messages().list(
        userId='me',
        q=full_query,
        maxResults=max_results
    ).execute()

    messages = results.get('messages', [])
    if not messages:
        return []

    emails = []
    for msg_ref in messages:
        msg = service.users().messages().get(
            userId='me',
            id=msg_ref['id'],
            format='full'
        ).execute()

        headers = {h['name'].lower(): h['value'] for h in msg['payload'].get('headers', [])}
        body = extract_body(msg['payload'])

        emails.append({
            'from': headers.get('from', ''),
            'subject': headers.get('subject', '(sans objet)'),
            'date': headers.get('date', ''),
            'snippet': msg.get('snippet', ''),
            'body': body[:2000],
        })

    print(f"DEBUG GMAIL: {len(emails)} emails récupérés", flush=True)
    return emails


def format_emails_for_context(emails):
    """Formate une liste d'emails en texte pour injection dans le prompt LLM."""
    if not emails:
        return "Aucun email trouvé pour cette recherche."

    blocks = []
    for i, e in enumerate(emails, 1):
        block = f"[Email {i}]\nDe: {e['from']}\nDate: {e['date']}\nSujet: {e['subject']}\n\nExtrait: {e['snippet']}"
        if e['body'] and len(e['body']) > len(e['snippet']):
            block += f"\n\nContenu:\n{e['body']}"
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)