"""Appointment scheduling simulation."""

from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class Appointment:
    patient_name: str
    specialty: str
    doctor: str
    date: str
    time: str
    reason: str
    notes: str = ""


class AppointmentScheduler:
    def __init__(self):
        self.appointments: list[Appointment] = []
        self._generate_available_slots()

    def _generate_available_slots(self):
        """Generate available time slots for the next 2 weeks."""
        self.available_slots: dict[str, list[str]] = {}
        today = datetime.now()

        for day_offset in range(1, 15):
            date = today + timedelta(days=day_offset)
            if date.weekday() >= 5:  # skip weekends
                continue
            date_str = date.strftime("%Y-%m-%d")
            self.available_slots[date_str] = [
                "08:00", "08:30", "09:00", "09:30", "10:00", "10:30",
                "11:00", "11:30", "12:00", "13:00", "13:30", "14:00",
                "14:30", "15:00", "15:30", "16:00", "16:30",
            ]

    def get_next_available(self, specialty: str = None, preferred_date: str = None) -> list[dict]:
        """Get next 3 available appointments."""
        results = []
        for date_str in sorted(self.available_slots.keys()):
            if preferred_date and date_str < preferred_date:
                continue
            for time_slot in self.available_slots[date_str]:
                if len(results) >= 3:
                    return results
                results.append({"date": date_str, "time": time_slot})
        return results

    def book_appointment(
        self,
        patient_name: str,
        specialty: str,
        doctor: str,
        date: str,
        time: str,
        reason: str,
    ) -> dict:
        """Book an appointment. Returns confirmation or error."""
        if date not in self.available_slots:
            return {"success": False, "error": "Brak dostępnych terminów w tym dniu."}

        if time not in self.available_slots[date]:
            return {"success": False, "error": f"Termin {time} jest już zajęty."}

        self.available_slots[date].remove(time)

        appointment = Appointment(
            patient_name=patient_name,
            specialty=specialty,
            doctor=doctor,
            date=date,
            time=time,
            reason=reason,
        )
        self.appointments.append(appointment)

        return {
            "success": True,
            "confirmation": (
                f"Wizyta umówiona: {specialty} u {doctor}, "
                f"dnia {date} o godzinie {time}. "
                f"Pacjent: {patient_name}."
            ),
        }

    def get_available_dates_text(self) -> str:
        """Get formatted available dates for the LLM."""
        dates = sorted(self.available_slots.keys())[:5]
        lines = []
        for d in dates:
            slots = self.available_slots[d][:5]
            lines.append(f"  {d}: {', '.join(slots)} ...")
        return "\n".join(lines)
