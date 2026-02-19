"""JavaScript snippets for Hextris DOM interaction.

Single source of truth for the JS code injected via Selenium
``execute_script()`` to detect game state, restart the game, read
score, and dispatch rotation input.

Hextris exposes key globals on ``window``:

- ``window.gameState`` — int: 0=start, 1=playing, 2=game_over,
  -1=paused, 3=fade_out, 4=main_menu
- ``window.score`` — int: current score
- ``window.MainHex`` — the central hexagon object
- ``window.blocks`` — array of active blocks
- ``window.settings`` — game settings (e.g. ``rows``, ``len``)
- ``window.rush`` — rush multiplier for block speed

These are gray-box signals used to make training feasible.
"""

# ---------------------------------------------------------------------------
# State detection
# ---------------------------------------------------------------------------

DETECT_STATE_JS = """
return (function() {
    var result = {state: "gameplay", details: {}};

    if (typeof gameState === 'undefined') {
        result.state = "unknown";
        return result;
    }

    var gs = gameState;
    result.details.gameState = gs;

    if (gs === 0) {
        result.state = "menu";
    } else if (gs === 1) {
        result.state = "gameplay";
    } else if (gs === 2) {
        result.state = "game_over";
    } else if (gs === -1) {
        result.state = "menu";
    } else if (gs === 3) {
        // Fade-out animation before game over
        result.state = "gameplay";
    } else if (gs === 4) {
        result.state = "menu";
    } else {
        result.state = "unknown";
    }

    return result;
})();
"""

# ---------------------------------------------------------------------------
# Game over -- restart the game
# ---------------------------------------------------------------------------

DISMISS_GAME_OVER_JS = """
return (function() {
    // Hide the game-over screen overlay
    var gameoverscreen = document.getElementById('gameoverscreen');
    if (gameoverscreen) {
        gameoverscreen.style.webkitTransition = 'none';
        gameoverscreen.style.transition = 'none';
        gameoverscreen.style.opacity = 0;
    }

    // Remove blur from canvas
    var canvas = document.getElementById('canvas');
    if (canvas) {
        canvas.classList.remove('blur');
    }

    // Reinitialise the game (init(1) starts a new game)
    if (typeof init === 'function') {
        init(1);
        return {action: "init_restart", text: ""};
    }

    return {action: "none", text: ""};
})();
"""

# ---------------------------------------------------------------------------
# Menu -- start the game from the start screen
# ---------------------------------------------------------------------------

START_GAME_JS = """
return (function() {
    if (typeof gameState === 'undefined') {
        return {action: "none"};
    }

    // From start screen (0), paused (-1), or main menu (4)
    if (gameState === 0 || gameState === -1 || gameState === 4) {
        if (typeof resumeGame === 'function') {
            resumeGame();
            return {action: "resume_game"};
        }
        // Fallback: click startBtn
        var startBtn = document.getElementById('startBtn');
        if (startBtn) {
            startBtn.click();
            return {action: "start_btn_click"};
        }
    }

    // From game over (2) -- need to reinit first
    if (gameState === 2) {
        if (typeof init === 'function') {
            init(1);
        }
        var gameoverscreen = document.getElementById('gameoverscreen');
        if (gameoverscreen) {
            gameoverscreen.style.opacity = 0;
        }
        var canvas = document.getElementById('canvas');
        if (canvas) {
            canvas.classList.remove('blur');
        }
        return {action: "restart_from_gameover"};
    }

    return {action: "none"};
})();
"""

# ---------------------------------------------------------------------------
# Input -- rotate the hexagon via JS
# ---------------------------------------------------------------------------

ROTATE_LEFT_JS = """
return (function() {
    if (typeof MainHex !== 'undefined' && MainHex &&
        typeof MainHex.rotate === 'function') {
        MainHex.rotate(-1);
        return {ok: true, direction: "left"};
    }
    return {ok: false, error: "MainHex not available"};
})();
"""

ROTATE_RIGHT_JS = """
return (function() {
    if (typeof MainHex !== 'undefined' && MainHex &&
        typeof MainHex.rotate === 'function') {
        MainHex.rotate(1);
        return {ok: true, direction: "right"};
    }
    return {ok: false, error: "MainHex not available"};
})();
"""

# ---------------------------------------------------------------------------
# Game state -- read score from window.score
# ---------------------------------------------------------------------------

READ_GAME_STATE_JS = """
return (function() {
    return {
        score: (typeof score !== 'undefined') ? score : 0,
        gameState: (typeof gameState !== 'undefined') ? gameState : -99,
        blockCount: (typeof blocks !== 'undefined' && blocks) ? blocks.length : 0,
        running: (typeof gameState !== 'undefined') && gameState === 1
    };
})();
"""

# ---------------------------------------------------------------------------
# Mute -- disable Hextris sounds
# ---------------------------------------------------------------------------

MUTE_JS = """
(function() {
    // Hextris uses howler.js or native Audio — override globally
    if (typeof Howl !== 'undefined') {
        Howler.mute(true);
    }
    // Also override Audio.prototype.play to no-op
    Audio.prototype.play = function() { return Promise.resolve(); };
})();
"""
