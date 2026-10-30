#!/usr/bin/env bash
# Copyright (C) 2026 Lenovo
# All Rights Reserved.
# Lenovo Confidential Restricted.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
QUEUE_SCRIPT="${REPO_ROOT}/scripts/gpu_job_queue.py"
PID_FILE="${REPO_ROOT}/.gpu_queue.pid"
LOCK_FILE="${REPO_ROOT}/.gpu_queue.lock"
LOG_FILE="${REPO_ROOT}/.gpu_queue_daemon.log"
TMUX_SESSION="traj_eval_gpu_queue"

ACTION="${1:-start}"
shift || true

case "${ACTION}" in
  start)
    if [[ -f "${LOCK_FILE}" ]]; then
      EXISTING_PID="$(cat "${LOCK_FILE}" 2>/dev/null || true)"
      if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
        echo "[gpu-queue] Already running (PID ${EXISTING_PID}). Log: ${LOG_FILE}"
        exit 0
      fi
    fi
    if command -v tmux >/dev/null 2>&1; then
      tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
      tmux new-session -d -s "${TMUX_SESSION}" "cd '${REPO_ROOT}' && '${PYTHON_BIN}' '${QUEUE_SCRIPT}' $* >> '${LOG_FILE}' 2>&1"
      sleep 1
      DAEMON_PID="$(cat "${LOCK_FILE}" 2>/dev/null || true)"
      if [[ -n "${DAEMON_PID}" ]]; then
        echo "${DAEMON_PID}" > "${PID_FILE}"
      fi
      echo "[gpu-queue] Started sequential GPU watcher daemon in tmux session '${TMUX_SESSION}' (PID ${DAEMON_PID:-pending})."
    else
      setsid nohup "${PYTHON_BIN}" "${QUEUE_SCRIPT}" "$@" >> "${LOG_FILE}" 2>&1 < /dev/null &
      DAEMON_PID=$!
      echo "${DAEMON_PID}" > "${PID_FILE}"
      echo "[gpu-queue] Started sequential GPU watcher daemon (PID ${DAEMON_PID})."
    fi
    echo "[gpu-queue] Log file: ${LOG_FILE}"
    ;;
  status)
    if [[ -f "${LOCK_FILE}" ]]; then
      EXISTING_PID="$(cat "${LOCK_FILE}" 2>/dev/null || true)"
      if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
        echo "[gpu-queue] Daemon status: RUNNING (PID ${EXISTING_PID})"
      else
        echo "[gpu-queue] Daemon status: STOPPED (stale lock file)"
      fi
    else
      echo "[gpu-queue] Daemon status: STOPPED"
    fi
    "${PYTHON_BIN}" "${QUEUE_SCRIPT}" --status
    if [[ -f "${LOG_FILE}" ]]; then
      echo ""
      echo "=== Recent daemon log (${LOG_FILE}) ==="
      tail -n 15 "${LOG_FILE}"
    fi
    ;;
  stop)
    if command -v tmux >/dev/null 2>&1; then
      tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
    fi
    for F in "${LOCK_FILE}" "${PID_FILE}"; do
      if [[ -f "${F}" ]]; then
        P="$(cat "${F}" 2>/dev/null || true)"
        if [[ -n "${P}" ]] && kill -0 "${P}" 2>/dev/null; then
          kill "${P}" 2>/dev/null || true
          echo "[gpu-queue] Stopped PID ${P}."
        fi
        rm -f "${F}"
      fi
    done
    ;;
  install-cron)
    CRON_CMD="*/10 * * * * cd ${REPO_ROOT} && bash ${REPO_ROOT}/scripts/start_gpu_queue.sh start >/dev/null 2>&1"
    (crontab -l 2>/dev/null | grep -v "start_gpu_queue.sh" || true; echo "${CRON_CMD}") | crontab -
    echo "[gpu-queue] Installed 10-minute cron watchdog for ${REPO_ROOT}/scripts/start_gpu_queue.sh"
    ;;
  *)
    echo "Usage: $0 {start|status|stop|install-cron} [extra gpu_job_queue.py args]" >&2
    exit 2
    ;;
esac
