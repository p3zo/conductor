#!/bin/bash

set -e

: "${SOURCE_BRANCH:=main}"
: "${TARGET_BRANCH:=pages}"
: "${ORIGIN:=origin}"

# Disable this if you've already run the build script.
: "${BUILD:=true}"

# Explode and deploy the contents of this directory.
EXPLODE=dist

# Adapted from:
# https://github.com/git/git/blob/8d530c4d64ffcc853889f7b385f554d53db375ed/git-sh-setup.sh#L207-L222
ensure_clean_working_tree() {
    local err=0
    if ! git diff-files --quiet --ignore-submodules; then
        echo >&2 "Cannot deploy: You have unstaged changes."
        err=1
    fi
    if ! git diff-index --cached --quiet --ignore-submodules HEAD -- ; then
        if [[ "$err" -eq 0 ]]; then
            echo >&2 "Cannot deploy: Your index contains uncommitted changes."
        else
            echo >&2 "Additionally, your index contains uncommitted changes."
        fi
        err=1
    fi
    if [[ "$err" -ne 0 ]]; then
        exit "$err"
    fi
}

ensure_no_stopships() {
    if git grep -n -i "STOPSHIP" -- './*' ':!deploy.sh'; then
        echo >&2 "Cannot deploy: STOPSHIP annotations remain."
        exit 1
    fi
}

ensure_up_to_date() {
    git fetch "$ORIGIN"
    if [ "$(git rev-parse "$ORIGIN/$SOURCE_BRANCH")" \
            != "$(git rev-parse "$SOURCE_BRANCH")" ]; then
        echo >&2 "Cannot deploy: not up to date with $ORIGIN/$SOURCE_BRANCH."
        exit 1
    fi
}

prompt() {
    >&2 printf '%s\n' "$1"
    >&2 printf 'yes/no> '
    read line
    if [[ "$line" != "yes" ]]; then
        >&2 printf '%s\n' "Aborting!"
    fi
    printf '%s' "$line"
}

if [[ "$(git rev-parse --abbrev-ref HEAD)" != "$SOURCE_BRANCH" ]]; then
    msg="You're not on '$SOURCE_BRANCH'. Do you want to go there now?"
    if [[ "$(prompt "$msg")" != "yes" ]]; then
        exit 0
    else
        git checkout "$SOURCE_BRANCH"
    fi
fi

ensure_clean_working_tree
ensure_no_stopships
ensure_up_to_date

SOURCE_COMMIT="$(git rev-parse HEAD)"
printf 'Preparing to deploy commit %s.\n' "$SOURCE_COMMIT"

if [[ "$BUILD" == "true" ]]; then
    yarn build
else
    printf '%s\n' "Warning: skipping build."
fi

# Assemble the deploy commit in a worktree of its own. Staging it from this
# checkout is what once force-added the whole of node_modules to the site.
STAGE_DIR="$(mktemp -d)"
printf 'Staging directory: %s\n' "$STAGE_DIR"

# The deploy branch always starts from the remote, so a deploy that was built
# but never pushed is rebuilt rather than pushed later by accident.
if git rev-parse --verify --quiet "$ORIGIN/$TARGET_BRANCH" > /dev/null; then
    git worktree add -B "$TARGET_BRANCH" "$STAGE_DIR" "$ORIGIN/$TARGET_BRANCH"
else
    git worktree add --orphan -b "$TARGET_BRANCH" "$STAGE_DIR"
fi

git -C "$STAGE_DIR" rm -r --quiet --ignore-unmatch .
cp -R "$EXPLODE"/. "$STAGE_DIR"
git -C "$STAGE_DIR" add --all
git -C "$STAGE_DIR" commit --allow-empty --no-verify -m "Deploy: $SOURCE_COMMIT"

printf 'Deploying %s files:\n' "$(git -C "$STAGE_DIR" ls-files | wc -l | tr -d ' ')"
git -C "$STAGE_DIR" ls-files | sed 's/^/    /'

echo
printf 'Please review the build output now---run:\n'
printf '    cd "%s" && python3 -m http.server\n' "$STAGE_DIR"
msg="Do you want to deploy?"
if [[ "$(prompt "$msg")" == "yes" ]]; then
    git -C "$STAGE_DIR" push "$ORIGIN" "$TARGET_BRANCH"
fi

git worktree remove --force "$STAGE_DIR"
