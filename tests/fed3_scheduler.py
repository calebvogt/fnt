"""The scheduler model: when an event fires, and when it deliberately does not.

Every check here is a rule an unattended two-week protocol depends on. A mode
switch that fires an hour early, twice, or not at all after a restart is a
ruined experiment that looks like a successful one in the log.
"""
import os
import sys
from datetime import datetime, time as time_of_day, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fnt.fed3 import fed_scheduler as sched

RESULTS = []


def check(label, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")


def event(**kwargs):
    kwargs.setdefault("target", sched.ALL_DEVICES)
    kwargs.setdefault("action", sched.ACTION_DISPENSE)
    return sched.ScheduledEvent(**kwargs)


def main():
    # A Wednesday, so weekday arithmetic has somewhere to go in both directions.
    now = datetime(2026, 9, 2, 14, 0, 0)

    print("--- an alarm never resolves to a moment already gone ---")
    check("a time still to come today stays today",
          sched.resolve_alarm("Today", time_of_day(18, 30), now)
          == datetime(2026, 9, 2, 18, 30))
    check("a time already past rolls to tomorrow",
          sched.resolve_alarm("Today", time_of_day(9, 0), now)
          == datetime(2026, 9, 3, 9, 0),
          "entered at 14:00, '09:00 today' must not fire immediately")
    check("the current minute counts as past",
          sched.resolve_alarm("Today", time_of_day(14, 0), now)
          == datetime(2026, 9, 3, 14, 0))
    check("tomorrow is tomorrow",
          sched.resolve_alarm("Tomorrow", time_of_day(9, 0), now)
          == datetime(2026, 9, 3, 9, 0))
    check("a later weekday lands this week",
          sched.resolve_alarm("Friday", time_of_day(9, 0), now)
          == datetime(2026, 9, 4, 9, 0))
    check("an earlier weekday lands next week",
          sched.resolve_alarm("Monday", time_of_day(9, 0), now)
          == datetime(2026, 9, 7, 9, 0))
    check("today's weekday, time passed, lands next week",
          sched.resolve_alarm("Wednesday", time_of_day(9, 0), now)
          == datetime(2026, 9, 9, 9, 0))
    check("today's weekday, time still to come, stays today",
          sched.resolve_alarm("Wednesday", time_of_day(18, 0), now)
          == datetime(2026, 9, 2, 18, 0))

    print("\n--- a delay of zero is rejected rather than fired ---")
    check("zero yields no time", sched.resolve_timer(0, time_of_day(0, 0), now)
          == (None, 0))
    check("a delay resolves forward",
          sched.resolve_timer(0, time_of_day(1, 30), now)
          == (datetime(2026, 9, 2, 15, 30), 5400))
    check("days are counted",
          sched.resolve_timer(2, time_of_day(0, 0, 30), now)
          == (datetime(2026, 9, 4, 14, 0, 30), 172830))

    print("\n--- due() runs what is ready and skips what is not ---")
    s = sched.Scheduler()
    ready = s.add(event(target_time=now - timedelta(seconds=5)))
    future = s.add(event(target_time=now + timedelta(hours=1)))
    off = s.add(event(target_time=now - timedelta(seconds=5), enabled=False))
    stale = s.add(event(target_time=now - timedelta(hours=3)))
    due = s.due(now)
    check("an event just past its time is due", ready in due)
    check("a future event is not", future not in due)
    check("a disabled event is not", off not in due)
    check("a disabled event is not marked missed either",
          off.status == sched.STATUS_PENDING, off.status)
    check("an event overdue past the grace period is missed",
          stale.status == sched.STATUS_MISSED and stale not in due)
    check("and says why", "FNT was not running" in stale.last_result,
          stale.last_result)
    check("an event inside the grace period still runs",
          s.due(now)[0] is ready if s.due(now) else False)

    print("\n--- completing an event re-arms it only if it repeats ---")
    s = sched.Scheduler()
    once = s.add(event(target_time=now, repeat=sched.REPEAT_NONE))
    s.complete(once, True, "sent")
    check("a one-shot is done", once.status == sched.STATUS_DONE)
    failed = s.add(event(target_time=now, repeat=sched.REPEAT_NONE))
    s.complete(failed, False, "device rejected it")
    check("a failure is recorded as such", failed.status == sched.STATUS_FAILED)
    check("with the reason kept", failed.last_result == "device rejected it")

    hourly = s.add(event(target_time=datetime.now() - timedelta(minutes=5),
                         repeat=sched.REPEAT_HOURLY))
    s.complete(hourly, True, "sent")
    check("an hourly event re-arms", hourly.status == sched.STATUS_PENDING)
    check("into the future, not the past", hourly.target_time > datetime.now())
    check("by one step, not more",
          hourly.target_time <= datetime.now() + timedelta(hours=1))

    # The case a crash produces: a daily event days behind must land on the next
    # real occurrence, not tomorrow-relative-to-whenever-it-last-ran.
    daily = s.add(event(target_time=datetime.now() - timedelta(days=3, hours=2),
                        repeat=sched.REPEAT_DAILY))
    s.complete(daily, True, "sent")
    check("a daily event days behind catches up to the future",
          daily.target_time > datetime.now())
    check("and keeps its time of day",
          daily.target_time.hour == (datetime.now() - timedelta(hours=2)).hour,
          daily.target_time.strftime("%d %b %H:%M"))

    print("\n--- events are addressed by id, not by row ---")
    s = sched.Scheduler()
    first = s.add(event(target_time=now + timedelta(hours=5)))
    second = s.add(event(target_time=now + timedelta(hours=1)))
    check("the list sorts by fire time", s.events == [second, first])
    check("removing by id removes the right one",
          s.remove(first.id) is first and s.events == [second])
    check("removing an unknown id is harmless", s.remove("nope") is None)
    check("get finds by id", s.get(second.id) is second)

    print("\n--- unfinished business survives clearing ---")
    s = sched.Scheduler()
    done = s.add(event(target_time=now, status=sched.STATUS_DONE))
    missed = s.add(event(target_time=now, status=sched.STATUS_MISSED))
    pending = s.add(event(target_time=now + timedelta(hours=1)))
    repeating = s.add(event(target_time=now, status=sched.STATUS_DONE,
                            repeat=sched.REPEAT_DAILY))
    s.clear_finished()
    check("finished one-shots are dropped",
          done not in s.events and missed not in s.events)
    check("pending events stay", pending in s.events)
    check("a repeating event is never cleared", repeating in s.events,
          "it will fire again")

    print("\n--- renaming a cage keeps its events attached ---")
    s = sched.Scheduler()
    mine = s.add(event(target="FED 3", target_time=now))
    other = s.add(event(target="FED 4", target_time=now))
    s.rename_target("FED 3", "Cage B")
    check("the renamed target follows", mine.target == "Cage B")
    check("other targets are untouched", other.target == "FED 4")

    print("\n--- a schedule survives a crash ---")
    s = sched.Scheduler()
    original = s.add(event(target="Cage B", action=sched.ACTION_SET_MODE,
                           params={"mode": "Fixed Ratio (FR)", "ratio": 5,
                                   "timeout": 0},
                           target_time=now + timedelta(hours=2),
                           repeat=sched.REPEAT_DAILY, enabled=False,
                           status=sched.STATUS_PENDING))
    restored = sched.Scheduler()
    restored.load(s.to_list())
    copy = restored.events[0]
    check("the id is stable across a restart", copy.id == original.id)
    check("the target survives", copy.target == "Cage B")
    check("the parameters survive", copy.params == original.params)
    check("the fire time survives", copy.target_time == original.target_time)
    check("the repeat survives", copy.repeat == sched.REPEAT_DAILY)
    check("a disabled event stays disabled", copy.enabled is False)
    check("loading nothing is not an error",
          sched.Scheduler().load(None) is None)

    print("\n--- what the table says ---")
    disabled = event(target_time=now + timedelta(hours=1), enabled=False)
    check("a disabled event reads Disabled",
          disabled.describe_trigger(now) == "Disabled")
    pending_ev = event(target_time=now + timedelta(hours=1, minutes=2))
    check("a pending event counts down",
          pending_ev.describe_trigger(now) == "in 01:02:00",
          pending_ev.describe_trigger(now))
    over_a_day = event(target_time=now + timedelta(days=2, hours=3))
    check("a multi-day countdown shows days",
          over_a_day.describe_trigger(now).startswith("in 2d "),
          over_a_day.describe_trigger(now))
    mode_ev = event(action=sched.ACTION_SET_MODE,
                    params={"mode": "Fixed Ratio (FR)", "ratio": 5, "timeout": 30},
                    target_time=now)
    described = mode_ev.describe_action()
    check("a mode action names only the fields that mode uses",
          "ratio 5" in described and "timeout" not in described, described)
    lights_ev = event(action=sched.ACTION_LIGHTS, params={"lights": True},
                      target_time=now)
    check("a lights action reads plainly",
          lights_ev.describe_action() == "Lights ON")

    form_checks()

    passed = sum(RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


def form_checks():
    """The scheduling form itself, which is where a wrong time gets entered.

    The live hardware run exercises the delay path end to end; the alarm path
    resolves against the wall clock and so is checked here instead.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtCore import QTime
    from PyQt5.QtWidgets import QApplication
    from fnt.fed3.fed_widgets import FEDTabWidget

    app = QApplication.instance() or QApplication([])
    tab = FEDTabWidget()
    try:
        print("\n--- the scheduling form resolves what it previews ---")
        tab.sched_kind_combo.setCurrentIndex(0)          # "At a time"
        tab.sched_day_combo.setCurrentText("Tomorrow")
        tab.sched_alarm_time.setTime(QTime(3, 30, 0))
        when, offset = tab._resolve_sched_time()
        expected = sched.resolve_alarm("Tomorrow", time_of_day(3, 30))
        check("an alarm resolves to the chosen day and time",
              when == expected and offset == 0, str(when))
        check("the preview names that moment",
              when.strftime("%H:%M:%S") in tab.sched_preview.text(),
              tab.sched_preview.text())

        tab.sched_day_combo.setCurrentText("Today")
        tab.sched_alarm_time.setTime(QTime(0, 0, 1))
        when, _ = tab._resolve_sched_time()
        check("a time already gone today is pushed to tomorrow, not fired",
              when > datetime.now(), str(when))

        tab.sched_kind_combo.setCurrentIndex(1)          # "After a delay"
        tab.sched_delay_days.setValue(1)
        tab.sched_delay_time.setTime(QTime(2, 0, 0))
        when, offset = tab._resolve_sched_time()
        check("a delay resolves to now plus the delay", offset == 26 * 3600,
              str(offset))

        tab.sched_delay_days.setValue(0)
        tab.sched_delay_time.setTime(QTime(0, 0, 0))
        when, offset = tab._resolve_sched_time()
        check("a zero delay resolves to nothing", when is None)
        check("and the preview says so rather than showing a time",
              "greater than zero" in tab.sched_preview.text(),
              tab.sched_preview.text())

        print("\n--- the form only offers the fields the action uses ---")
        tab.sched_action_combo.setCurrentText(sched.ACTION_SET_MODE)
        check("a mode action offers a mode", not tab.sched_mode_combo.isHidden())
        check("and hides the lights choice", tab.sched_lights_combo.isHidden())
        tab.sched_action_combo.setCurrentText(sched.ACTION_LIGHTS)
        check("a lights action offers the lights choice",
              not tab.sched_lights_combo.isHidden())
        check("and hides the mode", tab.sched_mode_combo.isHidden())
        tab.sched_action_combo.setCurrentText(sched.ACTION_DISPENSE)
        check("a dispense needs neither",
              tab.sched_mode_combo.isHidden()
              and tab.sched_lights_combo.isHidden())
    finally:
        tab.cleanup()


if __name__ == "__main__":
    sys.exit(main())
