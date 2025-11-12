/*--------------------
CUSTOMIZABLE SETTINGS - Modify these values for your desired style
--------------------*/
let settings = {
    // ====== WAVE SHAPE ======
    lines: 16,                     // 🎨 Number of waves (higher = more waves, 5-50)
    amplitudeX: 200,               // 🎨 Horizontal wave length/width (20-300)
    amplitudeY: 100,                // 🎨 Vertical wave height/amplitude (0-200)
    offsetX: 5,                   // 🎨 Horizontal offset between waves (-20-20)
    smoothness: 0.01,              // 🎨 Wave smoothness - lower = smoother (0.001-0.1)
    
    // ====== START COLOR (Top/Left Gradient) ======
    hueStartColor: 282,             // 🎨 Starting hue (0-360): Red=0, Yellow=60, Green=120, Cyan=180, Blue=240, Magenta=300
    saturationStartColor: 74,      // 🎨 Starting saturation (0-100): 0=gray, 100=pure color
    lightnessStartColor: 67,       // 🎨 Starting lightness (0-100): 0=black, 50=medium, 100=white
    
    // ====== END COLOR (Bottom/Right Gradient) ======
    hueEndColor: 237,              // 🎨 Ending hue (0-360)
    saturationEndColor: 88,       // 🎨 Ending saturation (0-100)
    lightnessEndColor: 14,          // 🎨 Ending lightness (0-100)
    
    // ====== WAVE STYLE ======
    fill: true,                    // 🎨 true = solid filled waves, false = outline only
    crazyness: false,              // 🎨 true = random wave patterns, false = geometric waves
    speed: .005                       // 🎨 Animation speed (0 = static, 0.001-0.01 = animated)
};

/*--------------------
VARIABLES - Do not modify these
--------------------*/
const canvas = document.getElementById('waveCanvas');
const ctx = canvas.getContext('2d');
let winW = window.innerWidth;
let winH = window.innerHeight;
let Colors = [];
let waveOffsets = [];
let animationTime = 0;

/*--------------------
INITIALIZATION - Sets up colors and wave offsets
--------------------*/
function init() {
    canvas.width = winW;
    canvas.height = winH;

    // Generate gradient colors from start to end
    let startColor = `hsl(${settings.hueStartColor}, ${settings.saturationStartColor}%, ${settings.lightnessStartColor}%)`;
    let endColor = `hsl(${settings.hueEndColor}, ${settings.saturationEndColor}%, ${settings.lightnessEndColor}%)`;
    Colors = chroma.scale([startColor, endColor]).mode('lch').colors(settings.lines + 2);

    // Initialize random offsets for each wave
    waveOffsets = [];
    for (let i = 0; i < settings.lines; i++) {
        waveOffsets.push(Math.random() * 2 * Math.PI);
    }
}

init();

/*--------------------
DRAW WAVES - Main animation loop
--------------------*/
function drawWaves(time) {
    ctx.clearRect(0, 0, winW, winH);

    // Draw each wave line
    for (let i = 0; i < settings.lines; i++) {
        const yOffset = (winH / settings.lines) * i;
        const color = Colors[i + 1];
        ctx.beginPath();

        // Draw points along the wave
        for (let x = 0; x <= winW; x += 1) {
            let y;
            if (settings.crazyness) {
                // Random wave pattern
                y = yOffset + Math.sin(x * settings.smoothness + time + waveOffsets[i]) * settings.amplitudeY * Math.random();
            } else {
                // Smooth geometric wave pattern
                y = yOffset + Math.sin(x * settings.smoothness + time + waveOffsets[i]) * settings.amplitudeY;
            }

            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }

        // Apply fill or stroke style
        if (settings.fill) {
            ctx.lineTo(winW, winH);
            ctx.lineTo(0, winH);
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();
        } else {
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }

    // Continue animation
    requestAnimationFrame(drawWaves.bind(null, time + settings.speed));
}

drawWaves(0);

/*--------------------
WINDOW RESIZE HANDLER - Redraw on window resize
--------------------*/
window.addEventListener('resize', () => {
    winW = window.innerWidth;
    winH = window.innerHeight;
    init();
});
