"""
Mavenlink Task Analysis Helpers

Provides functions to analyze Mavenlink task data for PM workflow insights.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO date string to datetime"""
    if not date_str:
        return None
    try:
        # Handle various ISO formats
        date_str = date_str.replace('Z', '+00:00')
        if '+' not in date_str and '-' not in date_str[-6:]:
            date_str += '+00:00'
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None


def is_overdue(task: Dict[str, Any]) -> bool:
    """Check if a task is overdue (past due date and not completed)"""
    if task.get('status') in ('completed', 'done', 'closed'):
        return False
    due_date = parse_date(task.get('due_date'))
    if not due_date:
        return False
    return due_date < datetime.now(timezone.utc)


def days_since(date_str: Optional[str]) -> Optional[int]:
    """Calculate days since a given date"""
    date = parse_date(date_str)
    if not date:
        return None
    delta = datetime.now(timezone.utc) - date
    return delta.days


def calculate_task_metrics(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive task metrics for PM workflow analysis"""
    if not tasks:
        return {
            'total_tasks': 0,
            'completed_tasks': 0,
            'completion_rate': 0,
            'overdue_tasks': 0,
            'overdue_rate': 0,
            'blocked_tasks': 0,
            'tasks_without_due_date': 0,
            'avg_days_to_complete': None,
            'oldest_incomplete_task_days': None,
            'recent_completions_7d': 0,
            'recent_completions_30d': 0,
        }

    total = len(tasks)
    completed = []
    overdue = []
    blocked = []
    no_due_date = []
    completion_times = []

    now = datetime.now(timezone.utc)
    seven_days_ago = now.replace(hour=0, minute=0, second=0, microsecond=0)
    thirty_days_ago = now.replace(hour=0, minute=0, second=0, microsecond=0)

    oldest_incomplete_days = 0
    recent_7d = 0
    recent_30d = 0

    for task in tasks:
        status = (task.get('status') or '').lower()

        # Check completion status
        if status in ('completed', 'done', 'closed'):
            completed.append(task)

            # Calculate completion time
            created = parse_date(task.get('created_at'))
            completed_at = parse_date(task.get('completed_at'))
            if created and completed_at:
                days_to_complete = (completed_at - created).days
                if days_to_complete >= 0:
                    completion_times.append(days_to_complete)

                # Check if completed recently
                days_since_completed = (now - completed_at).days
                if days_since_completed <= 7:
                    recent_7d += 1
                if days_since_completed <= 30:
                    recent_30d += 1
        else:
            # Check if overdue
            if is_overdue(task):
                overdue.append(task)

            # Check if blocked
            if status in ('blocked', 'on_hold', 'waiting'):
                blocked.append(task)

            # Track oldest incomplete task
            created = parse_date(task.get('created_at'))
            if created:
                task_age = (now - created).days
                if task_age > oldest_incomplete_days:
                    oldest_incomplete_days = task_age

        # Check for missing due dates
        if not task.get('due_date'):
            no_due_date.append(task)

    completed_count = len(completed)
    overdue_count = len(overdue)

    return {
        'total_tasks': total,
        'completed_tasks': completed_count,
        'completion_rate': round((completed_count / total * 100), 1) if total > 0 else 0,
        'incomplete_tasks': total - completed_count,
        'overdue_tasks': overdue_count,
        'overdue_rate': round((overdue_count / total * 100), 1) if total > 0 else 0,
        'blocked_tasks': len(blocked),
        'tasks_without_due_date': len(no_due_date),
        'avg_days_to_complete': round(sum(completion_times) / len(completion_times), 1) if completion_times else None,
        'oldest_incomplete_task_days': oldest_incomplete_days if oldest_incomplete_days > 0 else None,
        'recent_completions_7d': recent_7d,
        'recent_completions_30d': recent_30d,
    }


def calculate_time_entry_metrics(time_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate time tracking metrics for PM workflow analysis"""
    if not time_entries:
        return {
            'total_entries': 0,
            'total_hours': 0,
            'entries_last_7d': 0,
            'entries_last_30d': 0,
            'hours_last_7d': 0,
            'hours_last_30d': 0,
            'days_since_last_entry': None,
            'unique_users_logging': 0,
            'avg_hours_per_entry': 0,
        }

    now = datetime.now(timezone.utc)
    total_minutes = 0
    entries_7d = 0
    entries_30d = 0
    minutes_7d = 0
    minutes_30d = 0
    user_ids = set()
    last_entry_date = None

    for entry in time_entries:
        # Get hours/minutes
        minutes = entry.get('time_in_minutes') or (entry.get('hours', 0) * 60)
        total_minutes += minutes

        # Track user
        user_id = entry.get('user_id')
        if user_id:
            user_ids.add(user_id)

        # Check recency
        entry_date = parse_date(entry.get('date_performed') or entry.get('created_at'))
        if entry_date:
            if last_entry_date is None or entry_date > last_entry_date:
                last_entry_date = entry_date

            days_ago = (now - entry_date).days
            if days_ago <= 7:
                entries_7d += 1
                minutes_7d += minutes
            if days_ago <= 30:
                entries_30d += 1
                minutes_30d += minutes

    total_hours = round(total_minutes / 60, 1)
    total_entries = len(time_entries)

    return {
        'total_entries': total_entries,
        'total_hours': total_hours,
        'entries_last_7d': entries_7d,
        'entries_last_30d': entries_30d,
        'hours_last_7d': round(minutes_7d / 60, 1),
        'hours_last_30d': round(minutes_30d / 60, 1),
        'days_since_last_entry': (now - last_entry_date).days if last_entry_date else None,
        'unique_users_logging': len(user_ids),
        'avg_hours_per_entry': round(total_hours / total_entries, 1) if total_entries > 0 else 0,
    }


def format_tasks_for_analysis(tasks: List[Dict[str, Any]], limit: int = 20) -> str:
    """Format tasks for crew agent analysis"""
    if not tasks:
        return "No tasks available"

    lines = []

    # Sort: overdue first, then by due date
    def sort_key(t):
        is_overdue_task = is_overdue(t)
        due = parse_date(t.get('due_date'))
        status = (t.get('status') or '').lower()
        is_complete = status in ('completed', 'done', 'closed')
        # Sort order: overdue first, then incomplete by due date, then completed
        return (
            not is_overdue_task,  # Overdue first
            is_complete,  # Incomplete before complete
            due or datetime.max.replace(tzinfo=timezone.utc)  # By due date
        )

    sorted_tasks = sorted(tasks, key=sort_key)[:limit]

    for task in sorted_tasks:
        status = (task.get('status') or 'unknown').lower()
        title = task.get('title') or task.get('name') or 'Untitled'
        due_date = task.get('due_date')

        # Status indicators
        if status in ('completed', 'done', 'closed'):
            status_icon = "[DONE]"
        elif status in ('blocked', 'on_hold', 'waiting'):
            status_icon = "[BLOCKED]"
        elif is_overdue(task):
            status_icon = "[OVERDUE]"
        else:
            status_icon = "[OPEN]"

        # Format due date
        if due_date:
            due = parse_date(due_date)
            due_str = due.strftime('%Y-%m-%d') if due else due_date[:10]
        else:
            due_str = "No due date"

        lines.append(f"{status_icon} {title} (Due: {due_str})")

    if len(tasks) > limit:
        lines.append(f"... and {len(tasks) - limit} more tasks")

    return "\n".join(lines)


def identify_pm_workflow_issues(
    task_metrics: Dict[str, Any],
    time_metrics: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Identify PM workflow issues based on metrics"""
    issues = []

    # Task-based issues
    if task_metrics.get('overdue_rate', 0) > 20:
        issues.append(f"HIGH OVERDUE RATE: {task_metrics['overdue_rate']}% of tasks are overdue ({task_metrics['overdue_tasks']} tasks)")

    if task_metrics.get('blocked_tasks', 0) > 3:
        issues.append(f"BLOCKED TASKS: {task_metrics['blocked_tasks']} tasks are blocked/on-hold")

    if task_metrics.get('tasks_without_due_date', 0) > 5:
        issues.append(f"MISSING DUE DATES: {task_metrics['tasks_without_due_date']} tasks have no due date set")

    if task_metrics.get('recent_completions_7d', 0) == 0 and task_metrics.get('incomplete_tasks', 0) > 0:
        issues.append("NO RECENT PROGRESS: No tasks completed in the last 7 days")

    oldest_days = task_metrics.get('oldest_incomplete_task_days')
    if oldest_days and oldest_days > 60:
        issues.append(f"STALE TASK: Oldest incomplete task is {oldest_days} days old")

    # Time tracking issues
    if time_metrics:
        days_since = time_metrics.get('days_since_last_entry')
        if days_since and days_since > 7:
            issues.append(f"TIME TRACKING GAP: No time entries in {days_since} days")

        if time_metrics.get('entries_last_7d', 0) == 0 and time_metrics.get('total_entries', 0) > 0:
            issues.append("STALE TIME TRACKING: No time logged in the last week")

    return issues


def build_pm_workflow_context(
    tasks: List[Dict[str, Any]],
    time_entries: Optional[List[Dict[str, Any]]] = None,
    workspace_id: Optional[str] = None
) -> str:
    """Build comprehensive PM workflow context for crew analysis"""

    task_metrics = calculate_task_metrics(tasks)
    time_metrics = calculate_time_entry_metrics(time_entries) if time_entries else None
    issues = identify_pm_workflow_issues(task_metrics, time_metrics)

    context_parts = []

    # Header
    context_parts.append("=== MAVENLINK PROJECT DATA ===")
    if workspace_id:
        context_parts.append(f"Workspace ID: {workspace_id}")

    # Task Summary
    context_parts.append("\n--- TASK METRICS ---")
    context_parts.append(f"Total Tasks: {task_metrics['total_tasks']}")
    context_parts.append(f"Completed: {task_metrics['completed_tasks']} ({task_metrics['completion_rate']}%)")
    context_parts.append(f"Incomplete: {task_metrics['incomplete_tasks']}")
    context_parts.append(f"Overdue: {task_metrics['overdue_tasks']} ({task_metrics['overdue_rate']}%)")
    context_parts.append(f"Blocked: {task_metrics['blocked_tasks']}")
    context_parts.append(f"Missing Due Dates: {task_metrics['tasks_without_due_date']}")

    if task_metrics['avg_days_to_complete']:
        context_parts.append(f"Avg Days to Complete: {task_metrics['avg_days_to_complete']}")

    context_parts.append(f"Completed Last 7 Days: {task_metrics['recent_completions_7d']}")
    context_parts.append(f"Completed Last 30 Days: {task_metrics['recent_completions_30d']}")

    # Time Tracking Summary
    if time_metrics:
        context_parts.append("\n--- TIME TRACKING METRICS ---")
        context_parts.append(f"Total Time Entries: {time_metrics['total_entries']}")
        context_parts.append(f"Total Hours Logged: {time_metrics['total_hours']}")
        context_parts.append(f"Hours Last 7 Days: {time_metrics['hours_last_7d']}")
        context_parts.append(f"Hours Last 30 Days: {time_metrics['hours_last_30d']}")
        context_parts.append(f"Team Members Logging Time: {time_metrics['unique_users_logging']}")
        if time_metrics['days_since_last_entry'] is not None:
            context_parts.append(f"Days Since Last Entry: {time_metrics['days_since_last_entry']}")

    # Issues/Warnings
    if issues:
        context_parts.append("\n--- PM WORKFLOW WARNINGS ---")
        for issue in issues:
            context_parts.append(f"⚠️ {issue}")
    else:
        context_parts.append("\n--- PM WORKFLOW STATUS ---")
        context_parts.append("✅ No major workflow issues detected")

    # Task Details
    context_parts.append("\n--- TASK DETAILS ---")
    context_parts.append(format_tasks_for_analysis(tasks))

    return "\n".join(context_parts)
