use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use totp_rs::{Algorithm, Secret, TOTP};

const LOGIN_URL: &str =
    "https://apiconnect.angelone.in/rest/auth/angelbroking/user/v1/loginByPassword";

#[derive(Debug)]
struct AngelCredentials {
    client_id: String,
    password: String,
    totp_secret: String,
    api_key: String,
}

#[derive(Debug, Deserialize)]
struct AuthResponse {
    status: Option<bool>,
    data: Option<AuthData>,
}

#[derive(Debug, Deserialize)]
struct AuthData {
    #[serde(rename = "jwtToken")]
    jwt_token: Option<String>,
    #[serde(rename = "feedToken")]
    feed_token: Option<String>,
}

pub async fn get_jwt_token() -> Result<(String, String), String> {
    let credentials = read_credentials()?;
    let totp_code = generate_totp_code(&credentials.totp_secret)?;
    let client = Client::new();
    authenticate(&client, &credentials, &totp_code).await
}

pub fn get_api_key() -> Result<String, String> {
    read_env("ANGEL_API_KEY")
}

pub fn get_client_id() -> Result<String, String> {
    read_env("ANGEL_CLIENT_ID")
}

fn read_credentials() -> Result<AngelCredentials, String> {
    Ok(AngelCredentials {
        client_id: read_env("ANGEL_CLIENT_ID")?,
        password: read_env("ANGEL_PASSWORD")?,
        totp_secret: read_env("ANGEL_TOTP_SECRET")?,
        api_key: read_env("ANGEL_API_KEY")?,
    })
}

fn read_env(key: &str) -> Result<String, String> {
    match std::env::var(key) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        Ok(_) => Err(format!("missing {key}")),
        Err(error) => Err(format!("missing {key}: {error}")),
    }
}

fn generate_totp_code(secret: &str) -> Result<String, String> {
    let cleaned_secret = secret.trim().replace(' ', "").to_ascii_uppercase();
    let secret_bytes = Secret::Encoded(cleaned_secret)
        .to_bytes()
        .map_err(|error| error.to_string())?;
    let totp = TOTP::new_unchecked(Algorithm::SHA1, 6, 1, 30, secret_bytes);

    totp.generate_current().map_err(|error| error.to_string())
}

async fn authenticate(
    client: &Client,
    credentials: &AngelCredentials,
    totp_code: &str,
) -> Result<(String, String), String> {
    let response = client
        .post(LOGIN_URL)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .header("X-UserType", "USER")
        .header("X-SourceID", "WEB")
        .header("X-ClientLocalIP", "127.0.0.1")
        .header("X-ClientPublicIP", "127.0.0.1")
        .header("X-MACAddress", "00:00:00:00:00:00")
        .header("X-PrivateKey", &credentials.api_key)
        .json(&json!({
            "clientcode": credentials.client_id,
            "password": credentials.password,
            "totp": totp_code,
        }))
        .send()
        .await
        .map_err(|error| error.to_string())?;

    let status = response.status();
    let body = response.text().await.map_err(|error| error.to_string())?;
    if !status.is_success() {
        return Err(format!("auth failed ({status}): {body}"));
    }

    let auth: AuthResponse = serde_json::from_str(&body)
        .map_err(|error| format!("auth JSON parse failed: {error}; raw: {body}"))?;

    if auth.status == Some(false) {
        return Err(format!("Angel One rejected credentials: {body}"));
    }

    let data = auth
        .data
        .ok_or_else(|| format!("auth response missing data: {body}"))?;
    let jwt_token = data
        .jwt_token
        .filter(|token| !token.is_empty())
        .ok_or_else(|| format!("auth response missing jwtToken: {body}"))?;
    let feed_token = data
        .feed_token
        .filter(|token| !token.is_empty())
        .ok_or_else(|| format!("auth response missing feedToken: {body}"))?;

    Ok((jwt_token, feed_token))
}
