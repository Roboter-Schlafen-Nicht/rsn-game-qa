"""JavaScript snippets for shapez.io DOM interaction.

Single source of truth for the JS code injected via Selenium
``execute_script()`` to detect game state, navigate menus, read
game progress, dispatch building/input actions, and dismiss modals.

shapez.io architecture (dev build on port 3005):

- ``window.globalRoot`` -- GameRoot instance (available in dev mode)
- ``window.globalRoot.app`` -- Application instance
- ``window.globalRoot.hubGoals`` -- HubGoals (level, storedShapes)
- ``document.body.id`` -- current state (e.g. ``"state_InGameState"``)
- ``window.globalRoot.hud`` -- HUD parts (unlock notification, etc.)

These are gray-box signals used to make training feasible for a game
with no natural game-over and complex progression (26 levels + freeplay).
Pure pixel-based observation cannot provide reward signals for shape
delivery or level progression.  The dev build (port 3005 + localhost)
enables ``G_IS_DEV = true`` and exposes ``window.globalRoot`` for full
state access.  This is an intentional exception to the pixel-only
constraint, analogous to Breakout 71's ``READ_GAME_STATE_JS`` for
score/level/lives.
"""

# ---------------------------------------------------------------------------
# State detection
# ---------------------------------------------------------------------------

DETECT_STATE_JS = """
return (function() {
    var result = {state: "unknown", details: {}};

    var bodyId = document.body ? document.body.id : "";
    result.details.bodyId = bodyId;

    if (bodyId === "state_InGameState") {
        // Check for blocking overlays
        if (typeof globalRoot !== 'undefined' && globalRoot && globalRoot.hud) {
            var parts = globalRoot.hud.parts;
            // Unlock notification (level complete modal)
            if (parts && parts.unlockNotification &&
                parts.unlockNotification.visible) {
                result.state = "level_complete";
                return result;
            }
            // Settings menu open
            if (parts && parts.settingsMenu &&
                parts.settingsMenu.visible) {
                result.state = "settings";
                return result;
            }
            // Modal dialogs
            if (parts && parts.modalDialogs &&
                parts.modalDialogs.dialogStack &&
                parts.modalDialogs.dialogStack.length > 0) {
                result.state = "modal";
                return result;
            }
        }
        result.state = "gameplay";
    } else if (bodyId === "state_MainMenuState") {
        result.state = "main_menu";
    } else if (bodyId === "state_PreloadState") {
        result.state = "loading";
    } else if (bodyId === "") {
        result.state = "loading";
    } else {
        result.state = "menu";
        result.details.stateName = bodyId;
    }

    return result;
})();
"""

# ---------------------------------------------------------------------------
# Read game state (level, shapes delivered, entities, etc.)
# ---------------------------------------------------------------------------

READ_GAME_STATE_JS = """
return (function() {
    var defaults = {
        level: 0,
        shapesDelivered: 0,
        goalRequired: 0,
        goalProgress: 0.0,
        entityCount: 0,
        upgradeLevels: {},
        running: false,
        inGame: false
    };

    if (typeof globalRoot === 'undefined' || !globalRoot) {
        return defaults;
    }

    var result = {};

    // Hub goals (level progression)
    var hg = globalRoot.hubGoals;
    if (hg) {
        result.level = hg.level || 0;

        // Current goal progress
        var goal = hg.currentGoal;
        if (goal) {
            result.goalRequired = goal.required || 0;
            // storedShapes maps shape keys to counts
            var stored = hg.storedShapes;
            if (stored && goal.definition) {
                var key = goal.definition.getHash
                    ? goal.definition.getHash()
                    : "";
                result.shapesDelivered = (stored[key] || 0);
            } else {
                result.shapesDelivered = 0;
            }
            result.goalProgress = result.goalRequired > 0
                ? Math.min(result.shapesDelivered / result.goalRequired, 1.0)
                : 0.0;
        } else {
            result.shapesDelivered = 0;
            result.goalRequired = 0;
            result.goalProgress = 0.0;
        }

        // Upgrade levels
        result.upgradeLevels = {};
        if (hg.upgradeLevels) {
            for (var key in hg.upgradeLevels) {
                result.upgradeLevels[key] = hg.upgradeLevels[key];
            }
        }
    } else {
        result.level = 0;
        result.shapesDelivered = 0;
        result.goalRequired = 0;
        result.goalProgress = 0.0;
        result.upgradeLevels = {};
    }

    // Entity count (buildings placed)
    var map = globalRoot.map;
    if (map && map.entityMap_ && typeof map.entityMap_.size !== 'undefined') {
        result.entityCount = map.entityMap_.size;
    } else if (globalRoot.entityMgr && globalRoot.entityMgr.entities) {
        result.entityCount = globalRoot.entityMgr.entities.length;
    } else {
        result.entityCount = 0;
    }

    // Game running state
    var app = globalRoot.app;
    if (app && app.stateMgr) {
        var currentState = app.stateMgr.currentState;
        result.inGame = currentState &&
            currentState.key === "InGameState";
        result.running = result.inGame &&
            currentState.stage === "s10_gameRunning";
    } else {
        result.inGame = false;
        result.running = false;
    }

    return result;
})();
"""

# ---------------------------------------------------------------------------
# Start new game from main menu
# ---------------------------------------------------------------------------

START_NEW_GAME_JS = """
return (function() {
    // Click the "Play" button on the main menu
    var playBtn = document.querySelector("button.playButton");
    if (playBtn) {
        playBtn.click();

        // Wait briefly, then look for "New Game" in the dialog
        // The main menu shows a save picker; we need "New Game"
        setTimeout(function() {
            var newGameBtn = document.querySelector(
                ".dialogButton.newGameButton, button.newGame"
            );
            if (newGameBtn) newGameBtn.click();
        }, 500);

        return {action: "play_button_clicked"};
    }

    // Fallback: try to find any "New Game" button directly
    var newGameBtn = document.querySelector(
        ".dialogButton.newGameButton, button.newGame"
    );
    if (newGameBtn) {
        newGameBtn.click();
        return {action: "new_game_clicked"};
    }

    return {action: "none", error: "No play/new-game button found"};
})();
"""

# ---------------------------------------------------------------------------
# Dismiss level-complete unlock notification
# ---------------------------------------------------------------------------

DISMISS_UNLOCK_JS = """
return (function() {
    if (typeof globalRoot === 'undefined' || !globalRoot || !globalRoot.hud) {
        return {action: "none", error: "globalRoot not available"};
    }

    var parts = globalRoot.hud.parts;
    if (parts && parts.unlockNotification && parts.unlockNotification.visible) {
        // Click the "Next Level" / continue button
        var btn = document.querySelector(
            ".unlockNotification .mainButton, " +
            ".unlockNotification .ok"
        );
        if (btn) {
            btn.click();
            return {action: "unlock_dismissed"};
        }

        // Fallback: try to close via the component method
        if (typeof parts.unlockNotification.close === 'function') {
            parts.unlockNotification.close();
            return {action: "unlock_closed_programmatic"};
        }
    }

    return {action: "none", error: "No unlock notification visible"};
})();
"""

# ---------------------------------------------------------------------------
# Dismiss any modal dialog
# ---------------------------------------------------------------------------

DISMISS_MODAL_JS = """
return (function() {
    if (typeof globalRoot === 'undefined' || !globalRoot || !globalRoot.hud) {
        return {action: "none"};
    }

    var parts = globalRoot.hud.parts;
    if (parts && parts.modalDialogs &&
        parts.modalDialogs.dialogStack &&
        parts.modalDialogs.dialogStack.length > 0) {
        // Click the OK/close button on the topmost dialog
        var btn = document.querySelector(
            ".modalDialog .ok, .modalDialog .close, " +
            ".modalDialog .dialogButton"
        );
        if (btn) {
            btn.click();
            return {action: "modal_dismissed"};
        }
    }

    return {action: "none"};
})();
"""

# ---------------------------------------------------------------------------
# Close settings menu
# ---------------------------------------------------------------------------

CLOSE_SETTINGS_JS = """
return (function() {
    if (typeof globalRoot === 'undefined' || !globalRoot || !globalRoot.hud) {
        return {action: "none"};
    }

    var parts = globalRoot.hud.parts;
    if (parts && parts.settingsMenu && parts.settingsMenu.visible) {
        if (typeof parts.settingsMenu.close === 'function') {
            parts.settingsMenu.close();
            return {action: "settings_closed"};
        }
    }

    // Fallback: press Escape
    document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', code: 'Escape'}));
    return {action: "escape_pressed"};
})();
"""

# ---------------------------------------------------------------------------
# Configure game for training (disable tutorials, speed up, etc.)
# ---------------------------------------------------------------------------

SETUP_TRAINING_JS = """
return (function() {
    var actions = [];

    if (typeof globalRoot === 'undefined' || !globalRoot) {
        return {actions: actions, error: "globalRoot not available"};
    }

    // Disable tutorial hints
    var app = globalRoot.app;
    if (app && app.settings) {
        try {
            app.settings.updateSetting("offerHints", false);
            actions.push("tutorials_disabled");
        } catch(e) {}
    }

    return {actions: actions};
})();
"""

# ---------------------------------------------------------------------------
# Input actions -- building placement, keyboard shortcuts
# ---------------------------------------------------------------------------

# Select a building by number key (1-9, 0)
SELECT_BUILDING_JS = """
var _sel_args = arguments;
return (function(keyCode) {
    var key = String(keyCode);
    document.dispatchEvent(new KeyboardEvent('keydown', {
        key: key, code: 'Digit' + key, keyCode: 48 + parseInt(key)
    }));
    document.dispatchEvent(new KeyboardEvent('keyup', {
        key: key, code: 'Digit' + key, keyCode: 48 + parseInt(key)
    }));
    return {ok: true, key: key};
})(_sel_args[0]);
"""

# Rotate selected building
ROTATE_BUILDING_JS = """
return (function() {
    document.dispatchEvent(new KeyboardEvent('keydown', {
        key: 'r', code: 'KeyR', keyCode: 82
    }));
    document.dispatchEvent(new KeyboardEvent('keyup', {
        key: 'r', code: 'KeyR', keyCode: 82
    }));
    return {ok: true, action: "rotate"};
})();
"""

# Click at a specific canvas position (place building / interact)
CLICK_AT_JS = """
var _sel_args = arguments;
return (function(x, y) {
    var canvas = document.getElementById('ingame_Canvas');
    if (!canvas) {
        return {ok: false, error: "Canvas not found"};
    }
    var rect = canvas.getBoundingClientRect();
    var clientX = rect.left + x;
    var clientY = rect.top + y;

    canvas.dispatchEvent(new MouseEvent('mousedown', {
        clientX: clientX, clientY: clientY,
        button: 0, bubbles: true
    }));
    canvas.dispatchEvent(new MouseEvent('mouseup', {
        clientX: clientX, clientY: clientY,
        button: 0, bubbles: true
    }));
    return {ok: true, x: x, y: y};
})(_sel_args[0], _sel_args[1]);
"""

# Delete entity at position (right-click)
DELETE_AT_JS = """
var _sel_args = arguments;
return (function(x, y) {
    var canvas = document.getElementById('ingame_Canvas');
    if (!canvas) {
        return {ok: false, error: "Canvas not found"};
    }
    var rect = canvas.getBoundingClientRect();
    var clientX = rect.left + x;
    var clientY = rect.top + y;

    canvas.dispatchEvent(new MouseEvent('mousedown', {
        clientX: clientX, clientY: clientY,
        button: 2, bubbles: true
    }));
    canvas.dispatchEvent(new MouseEvent('mouseup', {
        clientX: clientX, clientY: clientY,
        button: 2, bubbles: true
    }));
    return {ok: true, x: x, y: y, action: "delete"};
})(_sel_args[0], _sel_args[1]);
"""

# Pan camera via keyboard (WASD)
PAN_CAMERA_JS = """
var _sel_args = arguments;
return (function(direction) {
    var keyMap = {
        'up': {key: 'w', code: 'KeyW', keyCode: 87},
        'down': {key: 's', code: 'KeyS', keyCode: 83},
        'left': {key: 'a', code: 'KeyA', keyCode: 65},
        'right': {key: 'd', code: 'KeyD', keyCode: 68}
    };
    var k = keyMap[direction];
    if (!k) return {ok: false, error: "Invalid direction"};

    document.dispatchEvent(new KeyboardEvent('keydown', k));
    setTimeout(function() {
        document.dispatchEvent(new KeyboardEvent('keyup', k));
    }, 100);
    return {ok: true, direction: direction};
})(_sel_args[0]);
"""

# Center camera on hub (Space)
CENTER_HUB_JS = """
return (function() {
    document.dispatchEvent(new KeyboardEvent('keydown', {
        key: ' ', code: 'Space', keyCode: 32
    }));
    document.dispatchEvent(new KeyboardEvent('keyup', {
        key: ' ', code: 'Space', keyCode: 32
    }));
    return {ok: true, action: "center_hub"};
})();
"""

# No-op action (do nothing)
NOOP_JS = """
return {ok: true, action: "noop"};
"""

# ---------------------------------------------------------------------------
# Mute -- disable shapez.io sounds
# ---------------------------------------------------------------------------

MUTE_JS = """
(function() {
    // Override Web Audio API
    var origAC = window.AudioContext || window.webkitAudioContext;
    if (origAC) {
        var proto = origAC.prototype;
        var origCreateGain = proto.createGain;
        proto.createGain = function() {
            var gain = origCreateGain.call(this);
            gain.gain.value = 0;
            return gain;
        };
    }

    // Override HTML5 Audio
    Audio.prototype.play = function() { return Promise.resolve(); };

    // shapez.io uses app.settings for sound
    if (typeof globalRoot !== 'undefined' && globalRoot && globalRoot.app) {
        try {
            globalRoot.app.settings.updateSetting("musicVolume", 0);
            globalRoot.app.settings.updateSetting("soundVolume", 0);
        } catch(e) {}
    }
})();
"""
