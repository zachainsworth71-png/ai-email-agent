#!/usr/bin/env python3
import os
import base64
import datetime
import time
import logging
from email import message_from_bytes
import re

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Silence verbose logging
for noisy in ['googleapiclient.discovery_cache', 'googleapiclient.http', 'oauth2client', 'httplib2', 'httpx', 'openai']:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Constants
GMAIL_SCOPES     = ['https://www.googleapis.com/auth/gmail.modify']
CALENDAR_SCOPES  = ['https://www.googleapis.com/auth/calendar.events']
OPENAI_MODEL     = 'gpt-4o-mini'
BATCH_SIZE       = 10
SLEEP_INTERVAL   = 300   # seconds between cycles
EMAIL_QUERY      = 'is:unread -in:spam -category:promotions'
MAX_AGE_DAYS     = 90    # skip messages older than 90 days

# Domains/keywords to skip
SKIP_DOMAINS         = {'redditmail.com', 'reddit.com', 'lists.reddit.com'}
SKIP_KEYWORDS        = {'automated', 'mail delivery subsystem', 'mailer-daemon', 'postmaster'}
SKIP_LOCAL_KEYWORDS  = {'mail','noreply','no-reply','no_reply','mailer-daemon','postmaster'}

# Initialize OpenAI key
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.max_retries = 0
if not openai.api_key:
    logger.error('OPENAI_API_KEY not set.')
    exit(1)


def authenticate_google(scopes, token_file, cred_file, service_name, version):
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, scopes)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(cred_file, scopes)
        creds = flow.run_local_server(port=0)
        with open(token_file, 'w') as f:
            f.write(creds.to_json())
    return build(service_name, version, credentials=creds)


def fetch_messages(service, query):
    resp = service.users().messages().list(userId='me', q=query).execute()
    return resp.get('messages', []) or []


def get_body(service, msg_id):
    raw = service.users().messages().get(userId='me', id=msg_id, format='raw').execute()['raw']
    data = base64.urlsafe_b64decode(raw)
    msg = message_from_bytes(data)
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            return part.get_payload(decode=True).decode(errors='ignore')
    return ''


def send_reply(service, thread_id, to_addr, subject, content):
    message = f"From: me\nTo: {to_addr}\nSubject: Re: {subject}\nIn-Reply-To: {thread_id}\n\n{content}"
    raw = base64.urlsafe_b64encode(message.encode()).decode()
    try:
        service.users().messages().send(userId='me', body={'raw': raw, 'threadId': thread_id}).execute()
        logger.info('Replied in thread %s', thread_id)
    except Exception as e:
        logger.warning('Failed to send reply: %s', e)


def classify_email(body):
    prompt = f"""You are a triage assistant. A customer email follows:\n\n{body}\n\nRespond EXACTLY one word: SCHEDULE, SUPPORT, or ESCALATE."""
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        intent = resp.choices[0].message.content.strip().upper()
        return intent if intent in {'SCHEDULE','SUPPORT','ESCALATE'} else 'SUPPORT'
    except Exception:
        return 'SUPPORT'


def extract_meeting_time(body):
    prompt = f"Extract the meeting date and time in ISO-8601 from this email. If none, reply NONE.\n\n{body}"
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        iso = resp.choices[0].message.content.strip()
        return None if iso == 'NONE' else iso
    except Exception:
        return None


def schedule_event(service, subject, iso_dt):
    dt = datetime.datetime.fromisoformat(iso_dt.rstrip('Z'))
    end = dt + datetime.timedelta(hours=1)
    event = {
        'summary': subject,
        'start': {'dateTime': dt.isoformat()+'Z', 'timeZone': 'UTC'},
        'end':   {'dateTime': end.isoformat()+'Z', 'timeZone': 'UTC'}
    }
    service.events().insert(calendarId='primary', body=event).execute()
    logger.info('Event scheduled at %s', iso_dt)


def process_batch(gmail, calendar):
    msgs = fetch_messages(gmail, EMAIL_QUERY)[:BATCH_SIZE]
    if not msgs:
        return False
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=MAX_AGE_DAYS)
    logger.info('Processing %d msgs', len(msgs))
    for m in msgs:
        full = gmail.users().messages().get(userId='me', id=m['id'], format='full').execute()
        ts = int(full.get('internalDate', '0')) / 1000
        msg_dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
        hdrs = full['payload']['headers']
        sender = next((h['value'] for h in hdrs if h['name']=='From'), '')
        match = re.search(r'<([^>]+)>', sender)
        addr = match.group(1) if match else sender
        local = addr.split('@')[0].lower()
        domain = addr.split('@')[-1].lower()
        # Skip old messages
        if msg_dt < cutoff:
            logger.info('Skipping old %s dated %s', addr, msg_dt.date())
            gmail.users().messages().modify(userId='me', id=m['id'], body={'removeLabelIds':['UNREAD']}).execute()
            continue
        # Skip unwanted senders
        if (domain in SKIP_DOMAINS or
            any(kw in local for kw in SKIP_LOCAL_KEYWORDS) or
            any(kw in sender.lower() for kw in SKIP_KEYWORDS)):
            logger.info('Skipping %s', sender)
            gmail.users().messages().modify(userId='me', id=m['id'], body={'removeLabelIds':['UNREAD']}).execute()
            continue
        body = get_body(gmail, m['id'])
        thread_id = full.get('threadId')
        subject = next((h['value'] for h in hdrs if h['name']=='Subject'), '')
        action = classify_email(body)
        logger.info('Msg %s -> %s', m['id'], action)
        if action == 'SCHEDULE':
            iso = extract_meeting_time(body)
            if iso:
                schedule_event(calendar, subject, iso)
            else:
                send_reply(gmail, thread_id, sender, subject, 'Please specify date and time.')
        elif action == 'SUPPORT':
            send_reply(gmail, thread_id, sender, subject, 'Thank you, will get back soon.')
        else:
            logger.info('Escalate %s', m['id'])
        gmail.users().messages().modify(userId='me', id=m['id'], body={'removeLabelIds':['UNREAD']}).execute()
    return True


def main():
    gmail = authenticate_google(GMAIL_SCOPES, 'token_gmail.json', 'credentials.json', 'gmail', 'v1')
    cal   = authenticate_google(CALENDAR_SCOPES, 'token_calendar.json', 'credentials.json', 'calendar', 'v3')
    while True:
        # Delete all messages in Spam each cycle
        spam = fetch_messages(gmail, 'in:spam')
        logger.info('Found %d spam msgs', len(spam))
        for m in spam:
            gmail.users().messages().delete(userId='me', id=m['id']).execute()
            logger.info('Deleted spam %s', m['id'])
        if not process_batch(gmail, cal):
            logger.info('No unread messages; sleeping %d seconds', SLEEP_INTERVAL)
        time.sleep(SLEEP_INTERVAL)

if __name__ == '__main__':
    main()
