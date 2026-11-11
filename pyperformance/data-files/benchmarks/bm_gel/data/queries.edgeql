# The queries compiled by the "gel_edgeql" benchmark, see schema.esdl
# for the schema.
#
# This is what an issue tracker application would run to show the issue
# details page (a large read query with a deep shape, computeds, aggregates
# and backlinks) and to act on it.  All of the queries involve access
# policies, and the DML ones also involve rewrites, triggers and defaults.


# -- Issue details page ------------------------------------------------------

with
    issue := (
        select Issue
        filter .team.key = <str>$team_key and .number = <int64>$number
    ),
select issue {
    id,
    identifier,
    title,
    body,
    status,
    priority,
    estimate,
    labels,
    due_at,
    created_at,
    updated_at,
    closed_at,
    is_open,
    is_overdue,
    is_blocked,
    comment_count,
    author: {
        id,
        full_name,
        avatar_url,
    },
    team: {
        id,
        key,
        name,
        color,
        open_issue_count,
    },
    project: {
        id,
        name,
        state,
        issue_count,
        total_estimate,
        progress := (
            round(
                100 * count((select .issues filter not .is_open))
                / math::abs(.issue_count)
            )
            if .issue_count > 0 else 0.0
        ),
    },
    parent: {
        id,
        identifier,
        title,
        status,
    },
    children: {
        id,
        identifier,
        title,
        status,
        priority,
        assignees: {
            id,
            full_name,
        } order by @position,
    } order by .status then .number,
    assignees: {
        id,
        full_name,
        avatar_url,
        open_issue_count,
        @position,
    } order by @position then .last_name,
    blocked_by: {
        id,
        identifier,
        title,
        status,
        team: {key},
    } filter .is_open,
    blocks := (
        select .<blocked_by[is Issue] {
            id,
            identifier,
            title,
        }
        order by .number
    ),
    comments: {
        id,
        body,
        created_at,
        edited_at,
        is_own := .author = global current_user,
        author: {
            id,
            full_name,
            avatar_url,
        },
        mentions: {
            id,
            full_name,
        },
    } order by .created_at,
    history: {
        kind,
        occurred_at,
        old_value,
        new_value,
        actor: {
            id,
            full_name,
        },
    } order by .occurred_at desc
      limit 50,
    watchers := (
        select distinct (
            issue.assignees union issue.author union issue.comments.author
        ) {
            id,
            full_name,
        }
        order by .full_name
    ),
};


# -- Post a comment ----------------------------------------------------------

with
    issue := assert_exists((select Issue filter .id = <uuid>$issue_id)),
    body := <str>$body,
    mentioned := (
        select User
        filter .email in array_unpack(<array<str>>$mentions)
    ),
    comment := (
        insert Comment {
            subject := issue,
            body := body,
            mentions := mentioned,
        }
    ),
select comment {
    id,
    body,
    created_at,
    author: {id, full_name, avatar_url},
    mentions: {id, full_name},
    issue := issue {
        id,
        identifier,
        comment_count,
    },
};


# -- Move issues to another status -------------------------------------------

with
    ids := <array<uuid>>$ids,
    new_status := <IssueStatus>$status,
    updated := (
        update Issue
        filter .id in array_unpack(ids) and .status != new_status
        set {
            status := new_status,
            assignees += (
                select global current_user {@position := 100}
                filter new_status = IssueStatus.InProgress
            ),
        }
    ),
select updated {
    id,
    identifier,
    status,
    closed_at,
    updated_at,
    is_blocked,
    assignees: {id, full_name} order by @position,
};
