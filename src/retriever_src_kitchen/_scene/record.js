/* Capture the real WebGL stream and rasterized live controls into a local video. */
let recording = null;
const recordButton = document.getElementById('record');
const recordStatus = document.getElementById('record-status');
const captureOptions = {scale:1,backgroundColor:null,logging:false,
  ignoreElements:el=>el.tagName==='IFRAME'||el.tagName==='VIDEO'};

async function beginRecording() {
  if (recording) { recording.stop(); return; }
  recordButton.disabled=true;
  recordStatus.textContent='Preparing';
  let stream,video;
  const tourMode = Boolean(state?.camera_tour);
  try {
    await command('restart'); await command('pause');
    await command('camera',{preset:tourMode?'Overview':'Agent'});
    const frame=document.querySelector('iframe');
    const source=[...frame.contentDocument.querySelectorAll('canvas')].find(c=>c.getContext('webgl2'));
    if (!source) throw Error('The 3D canvas is not ready');
    stream=source.captureStream(30);video=document.createElement('video');
    video.muted=true;video.playsInline=true;video.srcObject=stream;
    video.style.cssText='position:fixed;width:1px;height:1px;opacity:0;pointer-events:none';
    document.body.appendChild(video);
    let startupTimer;
    try {
      await Promise.race([video.play(),new Promise((_,reject)=>{
        startupTimer=setTimeout(()=>reject(Error('This browser did not start canvas recording. Use the rendered video export.')),8000);
      })]);
    }finally{clearTimeout(startupTimer);}
    const canvas=document.createElement('canvas');
    canvas.width=Math.floor(innerWidth/2)*2;canvas.height=Math.floor(innerHeight/2)*2;
    const ctx=canvas.getContext('2d',{willReadFrequently:true});
    const parts=['header','.panel','.scene-head','.scene-footer','.playback-badge'].map(s=>document.querySelector(s));
    const memory = document.getElementById('memory');
    if (memory && !document.querySelector('.panel').contains(memory)) parts.push(memory);
    let pictures=[],capturing=false,stopped=false,frames=0,minVariance=Infinity;
    const captureUI=async()=>{
      if(capturing||stopped)return;
      capturing=true;
      try {pictures=await Promise.all(parts.map(async element=>({
        picture:await html2canvas(element,captureOptions),rect:element.getBoundingClientRect()
      })));}finally{capturing=false;}
    };
    await captureUI();
    const mime=['video/mp4;codecs=avc1.42001E','video/webm;codecs=vp9','video/webm'].find(t=>MediaRecorder.isTypeSupported(t));
    if(!mime)throw Error('Video recording is unavailable in this browser');
    const output=canvas.captureStream(30), recorder=new MediaRecorder(output,{mimeType:mime,videoBitsPerSecond:8_000_000});
    const chunks=[];let doneAt=null,pausedOnce=false,closeCamera=false;
    const started=performance.now();
    recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
    const timer=setInterval(captureUI,240);
    function draw(){
      if(stopped)return;
      ctx.fillStyle='#fff';ctx.fillRect(0,0,canvas.width,canvas.height);
      const rect=frame.getBoundingClientRect();
      ctx.drawImage(video,rect.x,rect.y,rect.width,rect.height);
      if(frames%30===0){
        const p=ctx.getImageData(Math.floor(rect.x),Math.floor(rect.y),Math.floor(rect.width),Math.floor(rect.height)).data;
        let sum=0,squares=0,n=0;for(let i=0;i<p.length;i+=64){sum+=p[i];squares+=p[i]*p[i];n++;}
        minVariance=Math.min(minVariance,squares/n-(sum/n)**2);
      }
      pictures.forEach(({picture,rect:r})=>ctx.drawImage(picture,r.x,r.y,r.width,r.height));
      frames++;
      const elapsed=(performance.now()-started)/1000;
      recordStatus.textContent='Recording '+elapsed.toFixed(0)+'s';
      if(!tourMode&&elapsed>4&&!closeCamera){closeCamera=true;command('camera',{preset:'Robot'});}
      if(!tourMode&&state?.motion.seasoning_seconds>1&&!pausedOnce){pausedOnce=true;command('pause').then(()=>setTimeout(()=>command('resume'),1300));}
      if(state?.motion.finished&&(!tourMode||!state.camera_tour)&&doneAt===null)doneAt=performance.now();
      if(doneAt!==null&&performance.now()-doneAt>1800){stop();return;}
      requestAnimationFrame(draw);
    }
    function stop(){
      if(stopped)return;
      stopped=true;clearInterval(timer);recorder.stop();recordStatus.textContent='Saving';
      recordButton.disabled=true;
    }
    recorder.onstop=async()=>{
      try {
        stream.getTracks().forEach(t=>t.stop());output.getTracks().forEach(t=>t.stop());video.srcObject=null;video.remove();
        const blob=new Blob(chunks,{type:recorder.mimeType.split(';')[0]});
        const response=await fetch('/api/recording',{method:'POST',headers:{
          'Content-Type':blob.type,'X-Retriever-Token':window.controlToken,
          'X-Recording-Metadata':JSON.stringify({width:canvas.width,height:canvas.height,
            render_frames:frames,min_scene_pixel_variance:minVariance,motion:state?.motion})
        },body:blob});
        const result=await response.json();if(!response.ok)throw Error(result.error);
        recordStatus.textContent='Saved';recordStatus.title=result.file;
        document.getElementById('recording-result').textContent=result.file;
      }catch(error){recordStatus.textContent='Recording failed';document.getElementById('error').textContent=error.message;document.getElementById('error').hidden=false;}
      finally{recording=null;recordButton.disabled=false;recordButton.title='Record run';recordButton.setAttribute('aria-label','Record run');recordButton.classList.remove('recording');}
    };
    recording={stop};recordButton.disabled=false;recordButton.title='Stop recording';recordButton.setAttribute('aria-label','Stop recording');recordButton.classList.add('recording');
    recorder.start(1000);requestAnimationFrame(draw);
    setTimeout(()=>command(tourMode?'tour':'resume'),1000);
  }catch(error){
    stream?.getTracks().forEach(t=>t.stop());
    if(video){video.srcObject=null;video.remove();}
    recording=null;recordButton.disabled=false;recordStatus.textContent='Recording failed';document.getElementById('error').textContent=error.message;document.getElementById('error').hidden=false;
  }
}
recordButton.onclick=beginRecording;
