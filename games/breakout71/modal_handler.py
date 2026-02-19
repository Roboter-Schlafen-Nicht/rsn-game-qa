"""JavaScript snippets for Breakout 71 DOM interaction.

Single source of truth for the JS code injected via Selenium
``execute_script()`` to detect and dismiss game modals (game-over,
perk picker, menu), read game state, and dispatch input events.
All scripts and the environment import from here.

Breakout 71 is the development testbed and is treated as gray-box:
JS score reading, puck position, and modal handling are all allowed.
The real black-box validation games (Hextris, shapez.io) will NOT
use game-state JS injection.
"""

# ---------------------------------------------------------------------------
# State detection
# ---------------------------------------------------------------------------

DETECT_STATE_JS = """
return (function() {
    var result = {state: "gameplay", details: {}};

    var hasAlert = document.body.classList.contains('has-alert-open');
    var popup = document.getElementById('popup');
    var closeBtn = document.getElementById('close-modale');

    if (!hasAlert) {
        result.state = "gameplay";
        return result;
    }

    if (!popup) {
        result.state = "unknown";
        return result;
    }

    var popupText = popup.innerText || "";
    var buttons = popup.querySelectorAll('button');
    var buttonTexts = [];
    for (var i = 0; i < buttons.length; i++) {
        buttonTexts.push(buttons[i].innerText.trim());
    }
    result.details.buttonTexts = buttonTexts;

    var closeBtnVisible = false;
    if (closeBtn) {
        var style = window.getComputedStyle(closeBtn);
        closeBtnVisible = (style.display !== 'none'
                           && style.visibility !== 'hidden');
    }

    if (!closeBtnVisible && buttons.length >= 2) {
        result.state = "perk_picker";
        result.details.numPerks = buttons.length;
        return result;
    }

    if (closeBtnVisible) {
        var hasRestart = false;
        for (var j = 0; j < buttonTexts.length; j++) {
            var t = buttonTexts[j].toLowerCase();
            if (t.indexOf("new") >= 0 || t.indexOf("restart") >= 0 ||
                t.indexOf("run") >= 0 || t.indexOf("yes") >= 0 ||
                t.indexOf("again") >= 0) {
                hasRestart = true;
                break;
            }
        }
        if (hasRestart || popupText.toLowerCase().indexOf("game over") >= 0 ||
            popupText.toLowerCase().indexOf("score") >= 0) {
            result.state = "game_over";
        } else {
            result.state = "menu";
        }
        return result;
    }

    result.state = "unknown";
    return result;
})();
"""

# ---------------------------------------------------------------------------
# Perk picker -- click a random perk button
# ---------------------------------------------------------------------------

CLICK_PERK_JS = """
return (function() {
    var popup = document.getElementById('popup');
    if (!popup) return {clicked: -1, text: ""};
    var buttons = popup.querySelectorAll('button');
    if (buttons.length === 0) return {clicked: -1, text: ""};
    var idx = Math.floor(Math.random() * buttons.length);
    var text = buttons[idx].innerText.trim();
    buttons[idx].click();
    return {clicked: idx, text: text};
})();
"""

# ---------------------------------------------------------------------------
# Game over -- click restart button or close modal
# ---------------------------------------------------------------------------

DISMISS_GAME_OVER_JS = """
return (function() {
    // Prefer window.restart() (our automation hook) which correctly
    // re-assigns window.gameState to the module-scoped object.
    // Clicking the DOM restart button triggers the internal restart()
    // which leaves window.gameState pointing at a stale temporary.
    if (typeof window.restart === 'function') {
        // Close modal first so the DOM is clean
        var closeBtn = document.getElementById('close-modale');
        if (closeBtn) closeBtn.click();
        window.restart({});
        return {action: "window_restart", text: ""};
    }
    // Fallback: click DOM restart button (original behavior)
    var popup = document.getElementById('popup');
    if (popup) {
        var buttons = popup.querySelectorAll('button');
        for (var i = 0; i < buttons.length; i++) {
            var t = buttons[i].innerText.trim().toLowerCase();
            if (t.indexOf("new") >= 0 || t.indexOf("restart") >= 0 ||
                t.indexOf("run") >= 0 || t.indexOf("yes") >= 0 ||
                t.indexOf("again") >= 0) {
                buttons[i].click();
                return {action: "restart_button",
                        text: buttons[i].innerText.trim()};
            }
        }
    }
    var closeBtnFallback = document.getElementById('close-modale');
    if (closeBtnFallback) {
        closeBtnFallback.click();
        return {action: "close_button", text: ""};
    }
    return {action: "none", text: ""};
})();
"""

# ---------------------------------------------------------------------------
# Menu -- close button or Escape fallback
# ---------------------------------------------------------------------------

DISMISS_MENU_JS = """
return (function() {
    var closeBtn = document.getElementById('close-modale');
    if (closeBtn) {
        closeBtn.click();
        return {action: "close_button"};
    }
    document.dispatchEvent(
        new KeyboardEvent('keydown', {key: 'Escape', code: 'Escape'}));
    return {action: "escape_key"};
})();
"""

# ---------------------------------------------------------------------------
# Input -- dispatch mousemove event via JS (avoids ActionChains HTTP
# round-trip which costs ~270ms per call)
# ---------------------------------------------------------------------------

MOVE_MOUSE_JS = """
var _sel_args = arguments;
return (function(id, cx, cy) {
    var canvas = document.getElementById(id);
    if (!canvas) return {ok: false, error: "canvas not found"};
    var evt = new MouseEvent('mousemove', {
        clientX: cx,
        clientY: cy,
        bubbles: true,
        cancelable: true
    });
    canvas.dispatchEvent(evt);
    return {ok: true};
})(_sel_args[0], _sel_args[1], _sel_args[2]);
"""

# ---------------------------------------------------------------------------
# Game state -- read score, level, lives from window.gameState
# (gray-box: Breakout 71 testbed only)
# ---------------------------------------------------------------------------

READ_GAME_STATE_JS = """
return (function() {
    if (typeof gameState === 'undefined') {
        return {score: 0, level: 0, lives: 0, running: false};
    }
    return {
        score: gameState.score || 0,
        level: gameState.currentLevel || 0,
        lives: gameState.balls || 0,
        running: !!gameState.running
    };
})();
"""
