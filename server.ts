import "dotenv/config";
import express from "express";
import path from "path";
import fs from "fs";
import os from "os";
import multer from "multer";
import { GoogleGenAI, Modality } from "@google/genai";
import ffmpeg from "fluent-ffmpeg";
import ffmpegStatic from "ffmpeg-static";
import { createServer as createViteServer } from "vite";

ffmpeg.setFfmpegPath(ffmpegStatic!);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: "500mb" }));

// Error handler to ensure JSON responses on global errors 
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
    console.error(err.stack);
    res.status(500).json({ error: err.message || "Internal Server Error" });
});

// Configure multer
const tmpDir = os.tmpdir();
const upload = multer({ dest: path.join(tmpDir, "uploads") });

type ProcessState = {
  status: "processing" | "completed" | "error";
  step: "uploading" | "analyzing" | "translating" | "voiceover" | "merging" | "completed";
  progress: number; // 0-100 within current step
  resultUrl?: string;
  error?: string;
  message?: string;
};

const jobs = new Map<string, ProcessState>();

app.post("/api/process", upload.single("video"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No video file provided" });
  }

  const jobId = req.file.filename;
  const videoPath = req.file.path;
  const originalName = req.file.originalname;
  const mimeType = req.file.mimetype;
  const apiKey = req.body.apiKey;
  const startTimeVal = req.body.startTime;
  const endTimeVal = req.body.endTime;
  const voiceProfile = req.body.voiceProfile || "Puck";
  const voicePitchVal = req.body.voicePitch;
  const zoomLevelVal = req.body.zoomLevel;
  const aspectRatio = req.body.aspectRatio || "original";
  const mirrorVideoVal = req.body.mirrorVideo;
  const autoCutIntervalVal = req.body.autoCutInterval;
  
  const startTime = startTimeVal ? parseFloat(startTimeVal) : undefined;
  const endTime = endTimeVal ? parseFloat(endTimeVal) : undefined;
  const voicePitch = voicePitchVal ? parseInt(voicePitchVal, 10) : 0;
  const zoomLevel = zoomLevelVal ? parseInt(zoomLevelVal, 10) : 100;
  const mirrorVideo = mirrorVideoVal === 'true';
  const autoCutInterval = autoCutIntervalVal ? parseInt(autoCutIntervalVal, 10) : 3;

  jobs.set(jobId, { status: "processing", step: "analyzing", progress: 0 });

  res.json({ jobId });

  // Start background processing
  processVideo(jobId, videoPath, originalName, mimeType, apiKey, startTime, endTime, voiceProfile, voicePitch, zoomLevel, aspectRatio, mirrorVideo, autoCutInterval).catch((err) => {
    console.error("Error processing video:", err);
    jobs.set(jobId, { status: "error", step: "merging", progress: 0, error: err.message });
  });
});

app.get("/api/status/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: "Job not found" });
  }
  res.json(job);
});

app.get("/api/stream/:jobId", (req, res) => {
  const outputPath = path.join(tmpDir, `${req.params.jobId}_output.mp4`);
  if (!fs.existsSync(outputPath)) {
    return res.status(404).json({ error: "File not found" });
  }
  res.sendFile(outputPath); // sendFile supports Range requests for video players
});

app.get("/api/download/:jobId", (req, res) => {
  const outputPath = path.join(tmpDir, `${req.params.jobId}_output.mp4`);
  if (!fs.existsSync(outputPath)) {
    return res.status(404).json({ error: "File not found" });
  }
  res.download(outputPath, "Recap_AI_Result.mp4");
});

async function processVideo(jobId: string, videoPath: string, originalName: string, mimeType: string, userApiKey?: string, startTime?: number, endTime?: number, voiceProfile: string = "Puck", voicePitch: number = 0, zoomLevel: number = 100, aspectRatio: string = "original", mirrorVideo: boolean = false, autoCutInterval: number = 3) {
  try {
    const apiKey = userApiKey || process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("Gemini API Key is missing. Please provide it in the input box.");
    }

    const ai = new GoogleGenAI({
      apiKey: apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });

    let targetVideoPath = videoPath;
    
    // Trim video if specific times are provided
    updateStatus(jobId, "analyzing", 5);
    if (startTime !== undefined && endTime !== undefined && endTime > startTime && startTime >= 0) {
      console.log(`Trimming video from ${startTime} to ${endTime}`);
      const trimmedPath = path.join(tmpDir, `${jobId}_trimmed.mp4`);
      await new Promise<void>((resolve, reject) => {
        ffmpeg(videoPath)
          .setStartTime(startTime)
          .setDuration(endTime - startTime)
          .outputOptions(['-c copy'])
          .on('end', () => resolve())
          .on('error', (err) => reject(err))
          .save(trimmedPath);
      });
      targetVideoPath = trimmedPath;
    }

    // Step 1: Upload to Gemini (Analyzing & Translating)
    updateStatus(jobId, "analyzing", 20);
    
    // Helper for retryable operations
    const runWithRetry = async <T>(operation: () => Promise<T>, opName: string): Promise<T> => {
      let retries = 15;
      while (retries > 0) {
        let timeoutId: NodeJS.Timeout;
        try {
          const timeoutPromise = new Promise<T>((_, reject) => {
            timeoutId = setTimeout(() => reject(new Error("Timeout: The operation took too long. (90000ms)")), 90000);
          });
          const result = await Promise.race([operation(), timeoutPromise]);
          clearTimeout(timeoutId!);
          return result;
        } catch (err: any) {
          clearTimeout(timeoutId!);
          const status = err?.status ?? err?.error?.status;
          const code = err?.code ?? err?.error?.code;
          const msg = typeof err?.message === 'string' ? err.message : JSON.stringify(err);
          
          const isRateLimit = status === "RESOURCE_EXHAUSTED" || status === 429 || code === 429 || msg.includes("429");
          const isTempError = status === "UNAVAILABLE" || status === 503 || msg.includes("503") || msg.includes("high demand") || msg.includes("fetch failed") || msg.includes("Timeout");
              
          if (isRateLimit || isTempError) {
             let waitTime = 10000;
             if (isRateLimit) {
                 const match = msg.match(/retry in (\d+)(?:\.\d+)?s/);
                 if (match && match[1]) {
                     waitTime = (parseInt(match[1], 10) + 2) * 1000;
                 } else {
                     waitTime = 60000;
                 }
             }
             console.log(`[${opName}] unavailable/exhausted/timeout, retrying in ${waitTime/1000}s... (${retries} retries left)`);
             
             // Update UI status to show waiting
             const currentJob = jobs.get(jobId);
             if (currentJob) {
               updateStatus(jobId, currentJob.step, currentJob.progress, `Free Tier AI Quota ပြည့်နေပါသည်။ စက္ကန့် ${Math.ceil(waitTime/1000)} အကြာတွင် အလိုအလျောက် ပြန်လည်လုပ်ဆောင်ပါမည်...`);
             }

             retries--;
             if (retries === 0) throw err;
             await new Promise(r => setTimeout(r, waitTime));
             
             // Reset message after waiting
             if (currentJob) {
               updateStatus(jobId, currentJob.step, currentJob.progress);
             }
          } else {
             throw err;
          }
        }
      }
      throw new Error("Failed after retries");
    };

    const uploadedFile = await runWithRetry(() => ai.files.upload({
      file: targetVideoPath,
      config: {
        mimeType: mimeType || "video/mp4",
        displayName: originalName,
      }
    }), "uploadFile");
    
    let isFileActive = false;
    let pollCount = 0;
    while (!isFileActive && pollCount < 20) {
      const fileInfo = await ai.files.get({ name: uploadedFile.name });
      if (fileInfo.state === "ACTIVE") {
        isFileActive = true;
      } else if (fileInfo.state === "FAILED") {
        throw new Error("Video file processing failed in Gemini API.");
      } else {
        console.log(`Waiting for file to be active... Current state: ${fileInfo.state}`);
        await new Promise(r => setTimeout(r, 5000));
        pollCount++;
      }
    }
    if (!isFileActive) {
      throw new Error("Video file processing timed out. Please try again.");
    }
    
    // Helper for retries
    const generateContentWithRetry = async (modelName: string, contents: any, config?: any) => {
      return runWithRetry(() => ai.models.generateContent({
        model: modelName,
        contents,
        config
      }), `generateContent(${modelName})`);
    };

    updateStatus(jobId, "translating", 10);
    const recapRes = await generateContentWithRetry(
      "gemini-3.1-flash-lite",
      [
        { fileData: { fileUri: uploadedFile.uri, mimeType: uploadedFile.mimeType } },
        { text: "Watch this video and write a comprehensive, detailed movie recap script in Burmese language that is long enough to cover the entire duration of the video. Describe all important actions and scenes. The script MUST be spoken Burmese (Burmese script). Do not include any timestamps, markdown, English words, or speaker labels. Just the plain Burmese text that can be read aloud as a voiceover padding the full video time." }
      ]
    );
    
    const script = recapRes.text || "အခုမြင်တွေ့နေရတာကတော့ ရုပ်ရှင်ဇာတ်လမ်းတစ်ပုဒ်ပါ။";
    console.log("Generated Script:", script);
    
    // Step 2: Voiceover
    updateStatus(jobId, "voiceover", 10);
    
    const sentences = script.split('။').map(s => s.trim()).filter(s => s.length > 0);
    let chunks: string[] = [];
    let currentChunk = "";
    
    for (const sentence of sentences) {
      if ((currentChunk + sentence).length > 4000) {
        if (currentChunk) chunks.push(currentChunk + "။");
        currentChunk = sentence;
      } else {
        currentChunk += (currentChunk ? " " : "") + sentence;
      }
    }
    if (currentChunk) chunks.push(currentChunk + "။");
    if (chunks.length === 0) chunks = [script];
    
    console.log(`Processing ${chunks.length} TTS chunks...`);
    const audioBuffers: Buffer[] = [];
    
    // Process chunks with a concurrency limit of 1 to avoid rate limits
    const concurrencyLimit = 1;
    for (let i = 0; i < chunks.length; i += concurrencyLimit) {
      const batch = chunks.slice(i, i + concurrencyLimit);
      const batchBuffers = await Promise.all(batch.map(async (chunk, batchIndex) => {
        const index = i + batchIndex;
        console.log(`Generating TTS for chunk ${index + 1}/${chunks.length}...`);
        
        // Update progress before starting chunk
        const progress = 10 + Math.floor((index / chunks.length) * 80);
        updateStatus(jobId, "voiceover", progress);
        
        const chunkRes = await generateContentWithRetry(
          "gemini-3.1-flash-tts-preview",
          [{ parts: [{ text: `Say clearly in Burmese: ${chunk}` }] }],
          {
            responseModalities: [Modality.AUDIO],
            speechConfig: {
              voiceConfig: { prebuiltVoiceConfig: { voiceName: voiceProfile } }
            }
          }
        );
        
        const audioDataIndex = chunkRes.candidates?.[0]?.content?.parts?.findIndex((p: any) => p.inlineData?.data);
        const base64Audio = audioDataIndex !== undefined && audioDataIndex >= 0 
          ? chunkRes.candidates![0].content.parts![audioDataIndex].inlineData?.data 
          : null;
          
        if (!base64Audio) throw new Error("No audio returned for chunk " + index);
        return Buffer.from(base64Audio, "base64");
      }));
      audioBuffers.push(...batchBuffers);
    }
    
    const pcmPath = path.join(tmpDir, `${jobId}.pcm`);
    fs.writeFileSync(pcmPath, Buffer.concat(audioBuffers));
    
    // Step 3: Merging
    updateStatus(jobId, "merging", 10);
    const outputPath = path.join(tmpDir, `${jobId}_output.mp4`);
    
    // Apply voice pitch using asetrate and atempo if voicePitch is not 0
    let audioFilters: string[] = [];
    if (voicePitch !== 0) {
      // Mapping pitch to multiplier: -10 = 0.8, +10 = 1.2
      const multiplier = 1.0 + (voicePitch * 0.02);
      const newRate = Math.round(24000 * multiplier);
      const tempo = 1.0 / multiplier;
      audioFilters.push(`asetrate=${newRate},aresample=24000,atempo=${tempo}`);
    }
    audioFilters.push("apad"); // Pad with silence to ensure infinite length for -shortest
    
    // Apply video editing filters
    let complexFilters = [];
    let outputOptions = ['-c:a aac', '-shortest'];
    
    if (audioFilters.length > 0) {
      complexFilters.push(`[1:a:0]${audioFilters.join(',')}[adjusted_audio]`);
      outputOptions.push('-map [adjusted_audio]');
    } else {
      outputOptions.push('-map 1:a:0');
    }

    let currentVStream = "[0:v:0]";
    let vFilterIndex = 0;
    
    if (aspectRatio === '16:9') {
      complexFilters.push(`${currentVStream}scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2,split[v1_${vFilterIndex}][v2_${vFilterIndex}]`);
      complexFilters.push(`[v1_${vFilterIndex}]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=20:20[bg_${vFilterIndex}]`);
      complexFilters.push(`[v2_${vFilterIndex}]scale=1920:1080:force_original_aspect_ratio=decrease[fg_${vFilterIndex}]`);
      complexFilters.push(`[bg_${vFilterIndex}][fg_${vFilterIndex}]overlay=(W-w)/2:(H-h)/2[stream_${vFilterIndex}]`);
      currentVStream = `[stream_${vFilterIndex}]`;
      vFilterIndex++;
    } else if (aspectRatio === '9:16') {
      complexFilters.push(`${currentVStream}scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2,split[v1_${vFilterIndex}][v2_${vFilterIndex}]`);
      complexFilters.push(`[v1_${vFilterIndex}]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:20[bg_${vFilterIndex}]`);
      complexFilters.push(`[v2_${vFilterIndex}]scale=1080:1920:force_original_aspect_ratio=decrease[fg_${vFilterIndex}]`);
      complexFilters.push(`[bg_${vFilterIndex}][fg_${vFilterIndex}]overlay=(W-w)/2:(H-h)/2[stream_${vFilterIndex}]`);
      currentVStream = `[stream_${vFilterIndex}]`;
      vFilterIndex++;
    }

    if (mirrorVideo) {
      complexFilters.push(`${currentVStream}hflip[stream_${vFilterIndex}]`);
      currentVStream = `[stream_${vFilterIndex}]`;
      vFilterIndex++;
    }

    // Dynamic zoom and crop
    if (zoomLevel > 100) {
      const zRatio = zoomLevel / 100.0;
      complexFilters.push(`${currentVStream}crop=w='iw/${zRatio}':h='ih/${zRatio}':x='(iw-ow)/2+(iw-ow)/2*sin(trunc(t/${autoCutInterval})*211)':y='(ih-oh)/2+(ih-oh)/2*cos(trunc(t/${autoCutInterval})*113)',scale=w='trunc((iw*${zoomLevel}/100)/2)*2':h='trunc((ih*${zoomLevel}/100)/2)*2'[stream_${vFilterIndex}]`);
      currentVStream = `[stream_${vFilterIndex}]`;
      vFilterIndex++;
    }
    
    // Fallback if no filters were added but we need an even width/height!
    if (vFilterIndex === 0) {
      complexFilters.push(`${currentVStream}scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2[stream_${vFilterIndex}]`);
      currentVStream = `[stream_${vFilterIndex}]`;
      vFilterIndex++;
    }

    if (vFilterIndex > 0) {
      outputOptions.push(`-map ${currentVStream}`);
      outputOptions.push('-c:v libx264', '-pix_fmt yuv420p', '-preset ultrafast', '-crf 28');
    } else {
      outputOptions.push('-map 0:v:0');
      outputOptions.push('-c:v copy');
    }
    
    await new Promise<void>((resolve, reject) => {
      let command = ffmpeg(targetVideoPath)
        .input(pcmPath)
        .inputOptions(['-f s16le', '-ar 24000', '-ac 1']);

      if (complexFilters.length > 0) {
        command = command.complexFilter(complexFilters);
      }

      command = command.outputOptions(outputOptions);

      command.on('end', () => resolve())
        .on('error', (err) => reject(err))
        .save(outputPath);
    });
    
    // Cleanup
    if (fs.existsSync(videoPath)) fs.unlinkSync(videoPath);
    if (targetVideoPath !== videoPath && fs.existsSync(targetVideoPath)) fs.unlinkSync(targetVideoPath);
    if (fs.existsSync(pcmPath)) fs.unlinkSync(pcmPath);
    
    jobs.set(jobId, { status: "completed", step: "completed", progress: 100, resultUrl: `/api/download/${jobId}` });
  } catch (err: any) {
    console.error("Processing error:", err);
    jobs.set(jobId, { status: "error", step: "completed", progress: 0, error: err.message });
  }
}

function updateStatus(jobId: string, step: ProcessState["step"], progress: number, message?: string) {
  jobs.set(jobId, { status: "processing", step, progress, message });
}

// Vite Middleware for Dev, Static processing for Prod
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on port ${PORT}`);
  });
}

startServer();
