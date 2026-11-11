# The schema of a made-up issue tracker application used by the
# "gel_schema" and "gel_edgeql" benchmarks.
#
# The goal is to have a reasonably sized schema exercising most features
# of the Gel schema definition language: scalar types, abstract types and
# multiple inheritance, links with properties, computed pointers, constraints,
# indexes, functions, aliases, globals, access policies, rewrites and
# triggers.

module default {

    # -- Annotations ---------------------------------------------------------

    abstract annotation ui_label;
    abstract inheritable annotation pii;

    # -- Scalars -------------------------------------------------------------

    scalar type Role extending enum<Owner, Admin, Member, Guest>;
    scalar type IssueStatus extending
        enum<Backlog, Todo, InProgress, InReview, Done, Cancelled>;
    scalar type Priority extending enum<Urgent, High, Medium, Low, None_>;
    scalar type ProjectState extending enum<Planned, Active, Paused, Archived>;

    scalar type Email extending str {
        constraint max_len_value(254);
        constraint regexp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$');
    }

    scalar type HexColor extending str {
        constraint regexp(r'^#[0-9a-fA-F]{6}$');
    }

    scalar type Estimate extending int16 {
        constraint min_value(0);
        constraint max_value(100);
    }

    scalar type IssueNumber extending sequence;

    # -- Custom constraints --------------------------------------------------

    abstract constraint not_blank {
        errmessage := '{__subject__} must not be blank';
        using (len(str_trim(__subject__)) > 0);
    }

    abstract constraint title_like(max: std::int64) extending max_len_value {
        errmessage := '{__subject__} must be at most {max} characters long';
    }

    # -- Globals -------------------------------------------------------------

    global current_user_id: uuid;
    global current_workspace_id: uuid;
    required global is_superuser: bool {
        default := false;
    }

    global current_user := (
        select User filter .id = global current_user_id
    );

    global current_role := (
        select Membership
        filter
            .user.id = global current_user_id
            and .workspace.id = global current_workspace_id
    ).role;

    # -- Functions -----------------------------------------------------------

    function is_overdue(due: optional datetime, status: IssueStatus) -> bool
        using (
            (due < datetime_of_statement()) ?? false
            and status not in {IssueStatus.Done, IssueStatus.Cancelled}
        );

    # -- Abstract links ------------------------------------------------------

    abstract link ordered {
        position: int64 {
            default := 0;
        }
    }

    # -- Mixins --------------------------------------------------------------

    abstract type Timestamped {
        required created_at: datetime {
            default := datetime_of_statement();
            readonly := true;
        }
        required updated_at: datetime {
            default := datetime_of_statement();
            rewrite update using (datetime_of_statement());
        }

        index on (.created_at);
    }

    abstract type Authored {
        required author: User {
            default := global current_user;
            readonly := true;
            on target delete restrict;
        }
    }

    abstract type Named {
        required name: str {
            constraint not_blank;
            constraint title_like(120);
            annotation ui_label := 'Name';
        }
        description: str;

        index on (str_lower(.name));
    }

    abstract type WorkspaceScoped {
        required workspace: Workspace {
            on target delete delete source;
        }

        index on (.workspace);

        access policy workspace_members_read
            allow select
            using (
                global is_superuser
                or .workspace.id ?= global current_workspace_id
            );

        access policy workspace_members_write
            allow insert, update
            using (
                global is_superuser
                or (
                    .workspace.id ?= global current_workspace_id
                    and (global current_role ?? Role.Guest) != Role.Guest
                )
            ) {
                errmessage := 'Guests cannot modify workspace data';
            };

        access policy workspace_admins_delete
            allow delete
            using (
                global is_superuser
                or (
                    .workspace.id ?= global current_workspace_id
                    and (global current_role ?? Role.Guest)
                        in {Role.Owner, Role.Admin}
                )
            );
    }

    abstract type Commentable {
        multi comments := .<subject[is Comment];
        property comment_count := count(.comments);
    }

    # -- People --------------------------------------------------------------

    type User extending Timestamped {
        required email: Email {
            constraint exclusive;
            annotation pii := 'true';
        }
        required first_name: str {
            constraint not_blank;
            annotation pii := 'true';
        }
        required last_name: str {
            constraint not_blank;
            annotation pii := 'true';
        }
        property full_name := .first_name ++ ' ' ++ .last_name;
        avatar_url: str;
        required is_active: bool {
            default := true;
        }

        multi memberships := .<user[is Membership];
        multi assigned_issues := .<assignees[is Issue];
        property open_issue_count := count((
            select .assigned_issues filter .is_open
        ));

        index on ((.last_name, .first_name));

        access policy self_or_superuser_can_modify
            allow update, delete
            using (
                global is_superuser
                or .id ?= global current_user_id
            );
        access policy anyone_can_see
            allow select;
        access policy anyone_can_sign_up
            allow insert;
    }

    # -- Workspaces & teams --------------------------------------------------

    type Workspace extending Timestamped, Named {
        required slug: str {
            constraint exclusive;
            constraint regexp(r'^[a-z0-9]+(?:-[a-z0-9]+)*$');
        }
        required plan: billing::Plan;
        archived_at: datetime;
        multi memberships := .<workspace[is Membership];
        multi members := .memberships.user;
        property seats_left := .plan.max_seats - count(.memberships);

        constraint expression on (.archived_at >= .created_at) {
            errmessage := 'A workspace cannot be archived before it is created';
        }

        access policy members_can_see
            allow select
            using (
                global is_superuser
                or (global current_user in .members) ?? false
            );
        access policy owners_can_modify
            allow update, delete
            using (
                global is_superuser
                or (global current_role ?= Role.Owner)
            );
        access policy anyone_can_create
            allow insert;
    }

    type Membership extending Timestamped, WorkspaceScoped {
        required user: User {
            on target delete delete source;
        }
        required role: Role {
            default := Role.Member;
        }

        constraint exclusive on ((.workspace, .user));
    }

    type Team extending Timestamped, Named, WorkspaceScoped {
        required key: str {
            constraint regexp(r'^[A-Z][A-Z0-9]{1,5}$');
        }
        color: HexColor;
        multi members: User {
            on target delete allow;
            joined_at: datetime;
        }
        multi issues := .<team[is Issue];
        property open_issue_count := count((
            select .issues filter .is_open
        ));

        constraint exclusive on ((.workspace, .key));
    }

    # -- Projects & issues ---------------------------------------------------

    type Project extending
        Timestamped, Authored, Named, WorkspaceScoped, Commentable
    {
        required state: ProjectState {
            default := ProjectState.Planned;
        }
        start_date: cal::local_date;
        target_date: cal::local_date;
        multi issues := .<project[is Issue];
        property issue_count := count(.issues);
        property total_estimate := sum(.issues.estimate);

        constraint exclusive on ((.workspace, str_lower(.name)));
        constraint expression on (.target_date >= .start_date) {
            errmessage := 'The target date must not precede the start date';
        }
    }

    type Issue extending Timestamped, Authored, WorkspaceScoped, Commentable {
        required number: IssueNumber {
            default := sequence_next(introspect IssueNumber);
            readonly := true;
        }
        required title: str {
            constraint not_blank;
            constraint title_like(300);
        }
        body: str;
        required status: IssueStatus {
            default := IssueStatus.Backlog;
        }
        required priority: Priority {
            default := Priority.None_;
        }
        required team: Team {
            on target delete restrict;
        }
        project: Project {
            on target delete allow;
        }
        parent: Issue {
            on target delete allow;
        }
        multi assignees extending ordered: User {
            on target delete allow;
        }
        multi blocked_by: Issue {
            on target delete allow;
        }
        multi labels: str;
        estimate: Estimate;
        due_at: datetime;
        closed_at: datetime {
            rewrite insert, update using (
                datetime_of_statement()
                if __specified__.status and not __subject__.is_open
                else __subject__.closed_at
            );
        }
        deleted_at: datetime;

        property identifier := .team.key ++ '-' ++ <str>.number;
        property is_open :=
            .status not in {IssueStatus.Done, IssueStatus.Cancelled};
        property is_overdue := is_overdue(.due_at, .status);
        property is_blocked := exists (select .blocked_by filter .is_open);
        multi children := .<parent[is Issue];
        multi history := .<issue[is audit::IssueEvent];

        constraint exclusive on ((.team, .number));
        constraint expression on (.parent != __subject__) {
            errmessage := 'An issue cannot be its own parent';
        }

        index on ((.team, .status));
        index fts::index on (
            fts::with_options(.title, language := fts::Language.eng)
        );

        access policy hide_deleted
            deny select
            using (exists .deleted_at and not global is_superuser);

        trigger log_status_change after update for each
            when (__old__.status != __new__.status)
            do (
                insert audit::IssueEvent {
                    issue := __new__,
                    actor := global current_user,
                    kind := 'status',
                    old_value := <json>__old__.status,
                    new_value := <json>__new__.status,
                }
            );
    }

    type Comment extending Timestamped, Authored {
        required subject: Commentable {
            on target delete delete source;
        }
        required body: str {
            constraint not_blank;
        }
        multi mentions: User {
            on target delete allow;
        }
        edited_at: datetime {
            rewrite update using (
                datetime_of_statement()
                if __specified__.body
                else __old__.edited_at
            );
        }

        index on ((.subject, .created_at));

        access policy anyone_can_read_and_post
            allow select, insert;
        access policy authors_can_modify
            allow update, delete
            using (
                global is_superuser
                or .author.id ?= global current_user_id
            );
    }

    # -- Aliases -------------------------------------------------------------

    alias MyIssue := (
        select Issue
        filter global current_user in .assignees and .is_open
    );
}


module billing {

    scalar type Currency extending enum<USD, EUR, GBP, JPY>;

    scalar type Money extending decimal {
        constraint min_value(0.0n);
    }

    type Plan extending default::Named {
        required code: str {
            constraint exclusive;
        }
        required price: Money;
        required currency: Currency {
            default := Currency.USD;
        }
        required max_seats: int32 {
            constraint min_value(1);
            default := 5;
        }
        multi features: str;
    }

    type Customer extending default::Timestamped {
        required workspace: default::Workspace {
            constraint exclusive;
            on target delete delete source;
        }
        required billing_email: default::Email;
        address: tuple<line1: str, city: str, postal_code: str, country: str>;

        access policy workspace_admins
            allow all
            using (
                global default::is_superuser
                or (
                    .workspace.id ?= global default::current_workspace_id
                    and (global default::current_role ?? default::Role.Guest)
                        in {default::Role.Owner, default::Role.Admin}
                )
            );
    }
}


module audit {

    abstract type Event {
        required occurred_at: datetime {
            default := datetime_of_statement();
            readonly := true;
        }
        actor: default::User {
            on target delete allow;
        }
        required kind: str {
            constraint max_len_value(64);
        }
        old_value: json;
        new_value: json;

        index on (.occurred_at);

        access policy append_only
            allow select, insert;
        access policy superuser_can_purge
            allow delete
            using (global default::is_superuser);
    }

    type IssueEvent extending Event {
        required issue: default::Issue {
            on target delete delete source;
        }
    }
}
