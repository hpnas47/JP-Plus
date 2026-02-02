"""Notification system for email and macOS desktop notifications."""

import logging
import os
import smtplib
import subprocess
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class NotificationResult:
    """Result of a notification attempt."""

    success: bool
    method: str
    message: str
    error: Optional[str] = None


class Notifier:
    """
    Send notifications via email and macOS desktop.

    Supports:
    - Email via SMTP
    - macOS desktop notifications via osascript
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        notification_email: Optional[str] = None,
    ):
        """Initialize notifier with email settings.

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            smtp_user: SMTP username
            smtp_password: SMTP password
            notification_email: Email address to send notifications to
        """
        settings = get_settings()

        self.smtp_host = smtp_host or settings.smtp_host
        self.smtp_port = smtp_port or settings.smtp_port
        self.smtp_user = smtp_user or settings.smtp_user
        self.smtp_password = smtp_password or settings.smtp_password
        self.notification_email = notification_email or settings.notification_email

    def _can_send_email(self) -> bool:
        """Check if email configuration is available."""
        return all([
            self.smtp_host,
            self.smtp_port,
            self.smtp_user,
            self.smtp_password,
            self.notification_email,
        ])

    def send_email(
        self,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
    ) -> NotificationResult:
        """Send email notification.

        Args:
            subject: Email subject
            body: Plain text body
            html_body: Optional HTML body

        Returns:
            NotificationResult
        """
        if not self._can_send_email():
            return NotificationResult(
                success=False,
                method="email",
                message="Email not configured",
                error="Missing SMTP configuration in .env",
            )

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"] = self.notification_email

            # Attach plain text
            msg.attach(MIMEText(body, "plain"))

            # Attach HTML if provided
            if html_body:
                msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(
                    self.smtp_user,
                    self.notification_email,
                    msg.as_string(),
                )

            logger.info(f"Email sent to {self.notification_email}")
            return NotificationResult(
                success=True,
                method="email",
                message=f"Email sent to {self.notification_email}",
            )

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return NotificationResult(
                success=False,
                method="email",
                message="Failed to send email",
                error=str(e),
            )

    def send_macos_notification(
        self,
        title: str,
        message: str,
        sound: bool = True,
    ) -> NotificationResult:
        """Send macOS desktop notification using osascript.

        Args:
            title: Notification title
            message: Notification body
            sound: Whether to play sound

        Returns:
            NotificationResult
        """
        # Check if we're on macOS
        if os.uname().sysname != "Darwin":
            return NotificationResult(
                success=False,
                method="macos",
                message="Not running on macOS",
                error="macOS notifications only available on macOS",
            )

        try:
            # Escape special characters for AppleScript
            title = title.replace('"', '\\"')
            message = message.replace('"', '\\"')

            sound_str = 'sound name "default"' if sound else ""

            script = f'''
            display notification "{message}" with title "{title}" {sound_str}
            '''

            subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
            )

            logger.info("macOS notification sent")
            return NotificationResult(
                success=True,
                method="macos",
                message="Desktop notification sent",
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to send macOS notification: {e}")
            return NotificationResult(
                success=False,
                method="macos",
                message="Failed to send notification",
                error=str(e),
            )
        except Exception as e:
            logger.error(f"Unexpected error sending notification: {e}")
            return NotificationResult(
                success=False,
                method="macos",
                message="Unexpected error",
                error=str(e),
            )

    def notify_success(
        self,
        year: int,
        week: int,
        games_predicted: int,
        value_plays: int,
        report_path: Optional[Path] = None,
    ) -> list[NotificationResult]:
        """Send success notifications.

        Args:
            year: Season year
            week: Week number
            games_predicted: Number of games predicted
            value_plays: Number of value plays found
            report_path: Path to generated report

        Returns:
            List of NotificationResults
        """
        results = []

        # Compose messages
        title = f"CFB Predictions Ready - Week {week}"
        summary = (
            f"Week {week} predictions complete!\n\n"
            f"Games Predicted: {games_predicted}\n"
            f"Value Plays Found: {value_plays}\n"
        )

        if report_path:
            summary += f"\nReport: {report_path}"

        # Send macOS notification
        macos_result = self.send_macos_notification(
            title=title,
            message=f"{games_predicted} games, {value_plays} value plays",
        )
        results.append(macos_result)

        # Send email
        if self._can_send_email():
            html_body = f"""
            <html>
            <body>
            <h2>{title}</h2>
            <p><strong>Season:</strong> {year}</p>
            <p><strong>Games Predicted:</strong> {games_predicted}</p>
            <p><strong>Value Plays:</strong> {value_plays}</p>
            {f'<p><strong>Report:</strong> {report_path}</p>' if report_path else ''}
            </body>
            </html>
            """

            email_result = self.send_email(
                subject=title,
                body=summary,
                html_body=html_body,
            )
            results.append(email_result)

        return results

    def notify_failure(
        self,
        error_message: str,
        year: Optional[int] = None,
        week: Optional[int] = None,
    ) -> list[NotificationResult]:
        """Send failure notifications.

        Args:
            error_message: Description of the error
            year: Season year (optional)
            week: Week number (optional)

        Returns:
            List of NotificationResults
        """
        results = []

        context = ""
        if year and week:
            context = f" ({year} Week {week})"

        title = f"CFB Predictions Failed{context}"

        # Send macOS notification
        macos_result = self.send_macos_notification(
            title=title,
            message=error_message[:100],  # Truncate for notification
        )
        results.append(macos_result)

        # Send email with full error
        if self._can_send_email():
            email_result = self.send_email(
                subject=title,
                body=f"Error during prediction run:\n\n{error_message}",
            )
            results.append(email_result)

        return results

    def notify_data_wait(
        self,
        year: int,
        week: int,
        wait_minutes: int,
    ) -> NotificationResult:
        """Send notification about waiting for data.

        Args:
            year: Season year
            week: Week number
            wait_minutes: Minutes waited so far

        Returns:
            NotificationResult
        """
        return self.send_macos_notification(
            title="CFB Model - Waiting for Data",
            message=f"Week {week} data not ready. Waited {wait_minutes} min...",
            sound=False,
        )
