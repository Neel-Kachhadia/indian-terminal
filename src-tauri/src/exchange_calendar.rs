use std::{
    collections::{HashMap, HashSet},
    sync::OnceLock,
};

use chrono::{DateTime, Datelike, Duration, NaiveDate, NaiveTime, TimeZone, Utc, Weekday};
use chrono_tz::{Asia::Kolkata, Tz};
use serde::Deserialize;

const CALENDAR_JSON: &str = include_str!("../../contracts/exchange-calendar.json");
static CALENDAR: OnceLock<ExchangeCalendar> = OnceLock::new();

#[derive(Debug, Deserialize)]
struct CalendarDocument {
    schema_version: u32,
    metadata_version: String,
    exchange: String,
    timezone: String,
    normal_session: SessionDocument,
    weekly_expiry_weekday: u32,
    coverage: CoverageDocument,
    holidays: Vec<String>,
    #[serde(default)]
    special_sessions: Vec<SpecialSessionDocument>,
    #[serde(default)]
    contract_fixtures: Vec<ContractFixture>,
}

#[derive(Debug, Deserialize)]
struct SessionDocument {
    open: String,
    close: String,
}

#[derive(Debug, Deserialize)]
struct CoverageDocument {
    start: String,
    end: String,
}

#[derive(Debug, Deserialize)]
struct SpecialSessionDocument {
    date: String,
    session_type: String,
    special_session: bool,
    open: Option<String>,
    close: Option<String>,
    timing_status: String,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
struct SpecialSession {
    session_type: String,
    special_session: bool,
    open: Option<NaiveTime>,
    close: Option<NaiveTime>,
    timing_status: String,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Deserialize)]
struct ContractFixture {
    name: String,
    at_utc: String,
    session: String,
    next_weekly_expiry: String,
}

#[derive(Debug)]
pub struct ExchangeCalendar {
    timezone: Tz,
    open: NaiveTime,
    close: NaiveTime,
    weekly_expiry_weekday: u32,
    coverage_start: NaiveDate,
    coverage_end: NaiveDate,
    holidays: HashSet<NaiveDate>,
    special_sessions: HashMap<NaiveDate, SpecialSession>,
    #[cfg_attr(not(test), allow(dead_code))]
    fixtures: Vec<ContractFixture>,
}

impl ExchangeCalendar {
    fn parse() -> Result<Self, String> {
        let document: CalendarDocument = serde_json::from_str(CALENDAR_JSON)
            .map_err(|error| format!("exchange calendar JSON invalid: {error}"))?;
        if document.schema_version != 2
            || document.exchange != "NSE"
            || document.metadata_version.trim().is_empty()
        {
            return Err(String::from("unsupported exchange calendar contract"));
        }
        if document.timezone != "Asia/Kolkata" {
            return Err(format!(
                "unsupported exchange timezone: {}",
                document.timezone
            ));
        }
        let parse_date = |value: &str| {
            NaiveDate::parse_from_str(value, "%Y-%m-%d")
                .map_err(|error| format!("exchange calendar date {value} invalid: {error}"))
        };
        let parse_time = |value: &str| {
            NaiveTime::parse_from_str(value, "%H:%M")
                .map_err(|error| format!("exchange calendar time {value} invalid: {error}"))
        };
        let special_sessions = document
            .special_sessions
            .into_iter()
            .map(|session| {
                if !session.special_session
                    || !matches!(
                        session.session_type.as_str(),
                        "SPECIAL" | "MUHURAT" | "HALF_SESSION"
                    )
                    || session.timing_status.trim().is_empty()
                {
                    return Err(String::from("invalid exchange special-session metadata"));
                }
                let day = parse_date(&session.date)?;
                let open = session.open.as_deref().map(parse_time).transpose()?;
                let close = session.close.as_deref().map(parse_time).transpose()?;
                if open.is_some() != close.is_some() {
                    return Err(String::from("incomplete exchange special-session timing"));
                }
                Ok((
                    day,
                    SpecialSession {
                        session_type: session.session_type,
                        special_session: session.special_session,
                        open,
                        close,
                        timing_status: session.timing_status,
                    },
                ))
            })
            .collect::<Result<HashMap<_, _>, String>>()?;
        Ok(Self {
            timezone: Kolkata,
            open: parse_time(&document.normal_session.open)?,
            close: parse_time(&document.normal_session.close)?,
            weekly_expiry_weekday: document.weekly_expiry_weekday,
            coverage_start: parse_date(&document.coverage.start)?,
            coverage_end: parse_date(&document.coverage.end)?,
            holidays: document
                .holidays
                .iter()
                .map(|value| parse_date(value))
                .collect::<Result<HashSet<_>, _>>()?,
            special_sessions,
            fixtures: document.contract_fixtures,
        })
    }

    fn ensure_covered(&self, day: NaiveDate) -> Result<(), String> {
        if day < self.coverage_start || day > self.coverage_end {
            return Err(format!(
                "NSE calendar metadata does not cover {day} ({}..{})",
                self.coverage_start, self.coverage_end
            ));
        }
        Ok(())
    }

    pub fn is_trading_day(&self, day: NaiveDate) -> Result<bool, String> {
        self.ensure_covered(day)?;
        if self.special_sessions.contains_key(&day) {
            return Ok(true);
        }
        Ok(!matches!(day.weekday(), Weekday::Sat | Weekday::Sun) && !self.holidays.contains(&day))
    }

    fn session_bounds(&self, day: NaiveDate) -> Result<(NaiveTime, NaiveTime), String> {
        self.ensure_covered(day)?;
        if let Some(session) = self.special_sessions.get(&day) {
            return match (session.open, session.close) {
                (Some(open), Some(close)) => Ok((open, close)),
                _ => Err(format!("NSE special-session timing pending for {day}")),
            };
        }
        if !self.is_trading_day(day)? {
            return Err(format!("NSE session is closed for {day}"));
        }
        Ok((self.open, self.close))
    }

    fn previous_trading_day(&self, nominal: NaiveDate) -> Result<NaiveDate, String> {
        let mut candidate = nominal;
        loop {
            if self.is_trading_day(candidate)? {
                return Ok(candidate);
            }
            candidate -= Duration::days(1);
        }
    }

    pub fn session_state(&self, instant: DateTime<Utc>) -> Result<&'static str, String> {
        let local = instant.with_timezone(&self.timezone);
        if let Some(session) = self.special_sessions.get(&local.date_naive()) {
            if session.open.is_none() || session.close.is_none() {
                return Ok("SCHEDULE_PENDING");
            }
            let time = local.time();
            if time < session.open.expect("checked") {
                return Ok("PRE_OPEN");
            }
            if time < session.close.expect("checked") {
                return Ok("OPEN");
            }
            return Ok("CLOSED");
        }
        if !self.is_trading_day(local.date_naive())? {
            return Ok("CLOSED");
        }
        let time = local.time();
        if time < self.open {
            return Ok("PRE_OPEN");
        }
        if time < self.close {
            return Ok("OPEN");
        }
        Ok("CLOSED")
    }

    pub fn next_session_change_ms(&self, instant: DateTime<Utc>) -> Result<i64, String> {
        let local = instant.with_timezone(&self.timezone);
        let state = self.session_state(instant)?;
        let (day, time) = match state {
            "PRE_OPEN" => (
                local.date_naive(),
                self.session_bounds(local.date_naive())?.0,
            ),
            "OPEN" => (
                local.date_naive(),
                self.session_bounds(local.date_naive())?.1,
            ),
            "SCHEDULE_PENDING" => {
                return Err(format!(
                    "NSE special-session timing pending for {}",
                    local.date_naive()
                ));
            }
            _ => {
                let mut candidate = local.date_naive() + Duration::days(1);
                while !self.is_trading_day(candidate)? {
                    candidate += Duration::days(1);
                }
                let time = self.session_bounds(candidate)?.0;
                (candidate, time)
            }
        };
        self.timezone
            .from_local_datetime(&day.and_time(time))
            .single()
            .map(|value| value.timestamp_millis())
            .ok_or_else(|| String::from("exchange session boundary is ambiguous"))
    }

    pub fn next_weekly_expiry(&self, instant: DateTime<Utc>) -> Result<NaiveDate, String> {
        let local = instant.with_timezone(&self.timezone);
        let today = local.date_naive();
        self.ensure_covered(today)?;
        let current_weekday = today.weekday().num_days_from_monday();
        let days_until = (self.weekly_expiry_weekday + 7 - current_weekday) % 7;
        let mut nominal = today + Duration::days(i64::from(days_until));
        loop {
            let expiry = self.previous_trading_day(nominal)?;
            let expiry_cutoff = self.session_bounds(expiry)?.1;
            if expiry > today || (expiry == today && local.time() < expiry_cutoff) {
                return Ok(expiry);
            }
            nominal += Duration::days(7);
            self.ensure_covered(nominal)?;
        }
    }
}

pub fn exchange_calendar() -> Result<&'static ExchangeCalendar, String> {
    if let Some(calendar) = CALENDAR.get() {
        return Ok(calendar);
    }
    let parsed = ExchangeCalendar::parse()?;
    let _ = CALENDAR.set(parsed);
    CALENDAR
        .get()
        .ok_or_else(|| String::from("exchange calendar initialization failed"))
}

pub fn nearest_weekly_expiry_label(instant: DateTime<Utc>) -> Result<String, String> {
    Ok(exchange_calendar()?
        .next_weekly_expiry(instant)?
        .format("%d-%b-%Y")
        .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_matches_shared_contract_fixtures() {
        let calendar = exchange_calendar().unwrap();
        for fixture in &calendar.fixtures {
            let instant = DateTime::parse_from_rfc3339(&fixture.at_utc)
                .unwrap()
                .with_timezone(&Utc);
            assert_eq!(
                calendar.session_state(instant).unwrap(),
                fixture.session,
                "{} session",
                fixture.name
            );
            assert_eq!(
                calendar
                    .next_weekly_expiry(instant)
                    .unwrap()
                    .format("%d-%b-%Y")
                    .to_string(),
                fixture.next_weekly_expiry,
                "{} expiry",
                fixture.name
            );
        }
    }

    #[test]
    fn holiday_expiry_moves_to_previous_trading_day() {
        let calendar = exchange_calendar().unwrap();
        let nominal = NaiveDate::from_ymd_opt(2026, 3, 31).unwrap();
        assert_eq!(
            calendar.previous_trading_day(nominal).unwrap(),
            NaiveDate::from_ymd_opt(2026, 3, 30).unwrap()
        );
    }

    #[test]
    fn calendar_fails_closed_outside_verified_coverage() {
        let calendar = exchange_calendar().unwrap();
        let error = calendar
            .is_trading_day(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
            .unwrap_err();
        assert!(error.contains("does not cover"));
    }

    #[test]
    fn known_muhurat_date_is_special_not_weekend_closed() {
        let calendar = exchange_calendar().unwrap();
        let day = NaiveDate::from_ymd_opt(2026, 11, 8).unwrap();
        assert!(calendar.is_trading_day(day).unwrap());
        let session = calendar.special_sessions.get(&day).unwrap();
        assert_eq!(session.session_type, "MUHURAT");
        assert!(session.special_session);
        assert_eq!(session.timing_status, "PENDING_EXCHANGE_CIRCULAR");
        let instant = Utc.with_ymd_and_hms(2026, 11, 8, 6, 0, 0).unwrap();
        assert_eq!(calendar.session_state(instant).unwrap(), "SCHEDULE_PENDING");
        assert!(calendar.next_session_change_ms(instant).is_err());
    }

    #[test]
    fn special_session_times_control_transitions_and_expiry_cutoff() {
        let mut calendar = ExchangeCalendar::parse().unwrap();
        let day = NaiveDate::from_ymd_opt(2026, 8, 18).unwrap();
        calendar.special_sessions.insert(
            day,
            SpecialSession {
                session_type: String::from("SPECIAL"),
                special_session: true,
                open: NaiveTime::from_hms_opt(18, 0, 0),
                close: NaiveTime::from_hms_opt(19, 0, 0),
                timing_status: String::from("VERIFIED"),
            },
        );
        let before = Utc.with_ymd_and_hms(2026, 8, 18, 12, 0, 0).unwrap();
        let during = Utc.with_ymd_and_hms(2026, 8, 18, 12, 45, 0).unwrap();
        let close = Utc.with_ymd_and_hms(2026, 8, 18, 13, 30, 0).unwrap();

        assert_eq!(calendar.session_state(before).unwrap(), "PRE_OPEN");
        assert_eq!(
            calendar.next_session_change_ms(before).unwrap(),
            Utc.with_ymd_and_hms(2026, 8, 18, 12, 30, 0)
                .unwrap()
                .timestamp_millis()
        );
        assert_eq!(calendar.session_state(during).unwrap(), "OPEN");
        assert_eq!(
            calendar.next_session_change_ms(during).unwrap(),
            close.timestamp_millis()
        );
        assert_eq!(calendar.session_state(close).unwrap(), "CLOSED");
        assert_eq!(calendar.next_weekly_expiry(during).unwrap(), day);
        assert_eq!(
            calendar.next_weekly_expiry(close).unwrap(),
            NaiveDate::from_ymd_opt(2026, 8, 25).unwrap()
        );
    }
}
