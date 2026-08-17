"""
Client Google Calendar pour TARS. Lecture seule.
Réutilise l'authentification Google du client Gmail.
"""
import os
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from gmail_client import get_google_credentials


def get_calendar_service():
    """Retourne un service Calendar authentifié."""
    creds = get_google_credentials()
    return build('calendar', 'v3', credentials=creds)


def get_upcoming_events(days_ahead=7, max_results=20):
    """
    Récupère les événements à venir des N prochains jours.
    Retourne une liste de dicts avec : summary, start, end, location, description.
    """
    service = get_calendar_service()
    
    now = datetime.now(timezone.utc).isoformat()
    time_max = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).isoformat()
    
    print(f"DEBUG CALENDAR: recherche événements des {days_ahead} prochains jours", flush=True)
    
    try:
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
    except Exception as e:
        print(f"DEBUG CALENDAR: erreur ({e})", flush=True)
        return []
    
    events = events_result.get('items', [])
    if not events:
        return []
    
    formatted = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        end = event['end'].get('dateTime', event['end'].get('date'))
        formatted.append({
            'summary': event.get('summary', '(sans titre)'),
            'start': start,
            'end': end,
            'location': event.get('location', ''),
            'description': event.get('description', '')[:300],
        })
    
    print(f"DEBUG CALENDAR: {len(formatted)} événements récupérés", flush=True)
    return formatted


def format_events_for_context(events):
    """Formate une liste d'événements pour injection dans le prompt LLM."""
    if not events:
        return "Aucun événement dans les prochains jours."
    
    lines = []
    for i, e in enumerate(events, 1):
        # Formater la date de façon lisible
        start = e['start']
        if 'T' in start:
            # DateTime complet
            dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
            date_str = dt.strftime("%a %d %b, %Hh%M")
        else:
            # Journée entière
            dt = datetime.fromisoformat(start)
            date_str = dt.strftime("%a %d %b") + " (journée)"
        
        line = f"{i}. [{date_str}] {e['summary']}"
        if e['location']:
            line += f" @ {e['location'][:40]}"
        lines.append(line)
    
    return "\n".join(lines)