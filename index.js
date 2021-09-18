#!/usr/bin/env node
const faceapi = require("face-api.js");
const ffmpeg = require("ffmpeg-stream");
const child_process = require("child_process");
const fs = require("fs");
const path = require("path");
const cla = require("command-line-args");
const clu = require("command-line-usage");
const os = require("os");
const { Transform } = require("stream")
 
class ExtractFrames extends Transform {
  constructor(delimiter) {
    super({ readableObjectMode: true })
    this.delimiter = Buffer.from(delimiter, "hex")
    this.buffer = Buffer.alloc(0)
  }
 
  _transform(data, enc, cb) {
    // Add new data to buffer
    this.buffer = Buffer.concat([this.buffer, data])
    while (true) {
      const start = this.buffer.indexOf(this.delimiter)
      if (start < 0) break // there's no frame data at all
      const end = this.buffer.indexOf(this.delimiter, start + this.delimiter.length)
      if (end < 0) break // we haven't got the whole frame yet
      this.push(this.buffer.slice(start, end)) // emit a frame
      this.buffer = this.buffer.slice(end) // remove frame data from buffer
      if (start > 0) console.error(`Discarded ${start} bytes of invalid data`)
    }
    cb()
  }
}

const usage = clu([
    {
        header: "facecrop",
        content: "Face recognition-based video cropper."
    },
    {
        header: "Synopsis",
        content: `\
$ facecrop [--interval 6] [--output output.mp4] input.mp4
$ facecrop [-f 6] [-o output.mp4] -i input.mp4\
`
    },
    {
        header: "Options",
        optionList: [
            {
                name: "input",
                alias: "i",
                typeLabel: "{underline file}",
                description: "Input video to process.",
                defaultOption: true,
            },
            {
                name: "interval",
                alias: "f",
                typeLabel: "{underline number}",
                description: "Every nth frame to analyze with AI. Defaults to half of FPS.",
            },
            {
                name: "output",
                alias: "o",
                typeLabel: "{underline file}",
                description: "Output video. Defaults to [filename]_facecrop[.ext]"
            },
            {
                name: "disable-lerp",
                alias: "l",
                typeLabel: " ",
                description: "Disables interpolation between frames. May speed up final video rendering, but will cause footage's crops to feel choppy. This is useful if interval is 1, since lerp is unnecessary in that case."
            },
            {
                name: "codec",
                alias: "c",
                typeLabel: "{underline string}",
                description: "FFMPEG output video codec. {underline copy} uses the same codec as input file. Defaults to ffmpeg's default."
            }
        ]
    }
])

const options = cla([
    { name: "input", type: String, alias: "i", defaultOption: true },
    { name: "interval", type: Number, alias: "f" },
    { name: "output", type: String, alias: "o" },
    { name: "disable-lerp", type: Boolean, alias: "l" },
    { name: "codec", type: String, alias: "c" }
])

if (!options.input) {
    console.log(usage);
    process.exit(1);
}

const inputFile = options.input;

let fps = (() => {
    let tok = child_process.execSync("ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate " + inputFile).toString().trim().split("/");
    return Math.round(tok[0] / tok[1])
})();

const frameInterval = options.interval || Math.round(fps/2);
const interpolateBetweenFaces = !options["disable-lerp"];
const outputFile = options.output || (path.basename(inputFile, path.extname(inputFile)) + "_facecrop" + path.extname(inputFile));

console.log("Options");
console.log("Input file:", inputFile)
console.log("Output file:", outputFile);
console.log("Interval:", frameInterval);
console.log("Lerp:", interpolateBetweenFaces);

let resolution = child_process.execSync("ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of default=nw=1:nk=1 " + inputFile).toString().split("\n").slice(0, 2).map(x => parseInt(x));
let sidewaysResolution = [
    Math.floor(resolution[1] / 16 * 9),
    resolution[1],
];
console.log("Original resolution:", resolution.join("x"));
console.log("Target resolution:", sidewaysResolution.join("x"));

if (fs.existsSync(outputFile)) {
    fs.unlinkSync(outputFile);
}

let codec = child_process.execSync("ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 " + inputFile).toString().trim();

const videoCodec = options.codec == "copy" ? codec : options.codec;
console.log("Codec:", videoCodec);

const tf = require("@tensorflow/tfjs-node-gpu");

(async function() {
    
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, "models"));

    const converter = new ffmpeg.Converter();

    converter.createInputFromFile(inputFile);
    let i = 0;
    let crops = [];
    const defaultCrop = {
        x: 0,
        y: 0,
        width: sidewaysResolution[0],
        height: sidewaysResolution[1]
    };

    converter.createOutputStream({
        f: "rawvideo",
        vcodec: "png",
        vf: "select=not(mod(n\\," + frameInterval + ")),fps=" + (fps / frameInterval), // filter to only frameInterval-th frames
        threads: os.cpus().length,
    })
        .pipe(new ExtractFrames("89504E470D0A1A0A")) // PNG magic
        .on("data", async frame => {
            let tensor = tf.node.decodePng(frame);
            let detection = await faceapi.detectSingleFace(tensor);
            let crop;
            if (!detection) {
                crop = crops[crops.length - 1] || defaultCrop;
            } else {
                let box = {
                    x: Math.floor(detection.box.x),
                    y: Math.floor(detection.box.y),
                    width: Math.floor(detection.box.width),
                    height: Math.floor(detection.box.height),
    
                    xCenter: 0,
                    yCenter: 0
                }
    
                box.xCenter = box.x + Math.floor(box.width / 2);
                box.yCenter = box.y + Math.floor(box.height / 2);
    
                crop = {
                    x: box.xCenter - Math.floor(sidewaysResolution[0] / 2),
                    y: 0,
                    width: sidewaysResolution[0],
                    height: sidewaysResolution[1]
                }
            }

            crops.push(crop);
            tensor.dispose();

            if (++i % 10 == 0) {
                process.stdout.write(".");
            }
        })
        .on("end", () => ended = true)

    await converter.run();

    console.log("done");

    let frameValues = [];

    if (interpolateBetweenFaces) {
        frameValues.push(['eq(n,0)', crops[0].x]);

        for (let i = 1; i < crops.length; i++) {
            let f = (i-1) * frameInterval;
            frameValues.push([
                i == crops.length - 1 ? (`gte(n,${f + 1})`) : (`between(n,${f+1},${f+frameInterval})`),
                `lerp(${crops[i-1].x},${crops[i].x},(mod(n-1,${frameInterval})+1)/${frameInterval})`
            ]);
        }
    } else {
        for (let i = 0; i < crops.length; i++) {
            let f = i * frameInterval;
            frameValues.push([
                i == crops.length - 1 ? (`gte(n,${f})`) : (`between(n,${f},${f+frameInterval-1})`),
                crops[i].x
            ]);
        }
    }

    let filterScript = `\
crop=${sidewaysResolution[0]}:${sidewaysResolution[1]}:'
${frameValues.map(x => `if(${x[0]},st(0,${x[1]}));`).join("\n")}
ld(0)
':0`;

    let ffmproc = child_process.execFile("ffmpeg", [
        "-i", inputFile,
        "-acodec", "copy",
        "-filter:v:0", filterScript,
        ...(videoCodec ? ["-vcodec", videoCodec] : []),
        "-threads", os.cpus().length,
        outputFile
    ]);

    ffmproc.stdout.pipe(process.stdout);
    ffmproc.stderr.pipe(process.stderr);
    process.stdin.pipe(ffmproc.stdin);

    ffmproc.on("exit", (code) => process.exit(code));
})();