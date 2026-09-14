import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';
import * as GaussianSplats3D from '/vendor/gaussian-splats-3d.module.js';

const $ = id => document.getElementById(id);
let runs = [], selected = null, loaded = null, currentView = 'model', framesLoaded = null;
let scene, renderer, camera, controls, cloud, pathGroup, grid, home, surface, splats, gaussianViewer, axes, inferredSurface;
let cloudLoading = null;
let runListSignature = '', detailsSignature = '';
const number = x => Number(x).toLocaleString();
async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) { const data = await response.json().catch(() => ({})); throw new Error(data.detail || `Request failed (${response.status})`); }
  return response.json();
}
function notify(text) { $('toast').textContent = text; $('toast').hidden = false; setTimeout(() => $('toast').hidden = true, 5000); }
function message(title, text) {
  $('modelMessage').hidden = false;
  $('modelMessage').querySelector('h2').textContent = title;
  $('modelMessage').querySelector('p').textContent = text;
}
function setupScene() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(48, 1, .01, 10000);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  $('canvasHost').appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotateSpeed = .65;
  camera.position.set(12, 10, 18);
  grid = new THREE.GridHelper(30, 30, 0x345b62, 0x263e48);
  grid.material.transparent = true; grid.material.opacity = .5;
  scene.add(grid);
  scene.add(new THREE.HemisphereLight(0xd9fff3,0x173039,2.2));
  const sun=new THREE.DirectionalLight(0xffffff,2.4);sun.position.set(7,12,5);scene.add(sun);
  const fill=new THREE.DirectionalLight(0x78c7ff,.8);fill.position.set(-8,5,-6);scene.add(fill);
  axes=new THREE.AxesHelper(3);axes.material.depthTest=false;axes.renderOrder=10;scene.add(axes);
  const resize = () => {
    const {width, height} = $('canvasHost').getBoundingClientRect();
    if (!width || !height) return;
    renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe($('canvasHost')); resize();
  renderer.setAnimationLoop(() => { if (currentView === 'model') { controls.update();if(gaussianViewer){gaussianViewer.update();gaussianViewer.render();}else renderer.render(scene, camera); } });
}
function dispose(object) {
  if (!object) return;
  object.traverse(child => { child.geometry?.dispose(); if (Array.isArray(child.material)) child.material.forEach(m => m.dispose()); else child.material?.dispose(); });
  scene.remove(object);
}
function fitView() {
  if (!home || !controls) return;
  controls.target.copy(home.target); camera.position.copy(home.position); camera.up.copy(home.up||new THREE.Vector3(0,1,0)); controls.update();
}
async function loadCloud(id) {
  if (loaded === id || cloudLoading === id) return;
  cloudLoading = id;
  message('Opening your 3D model', 'Loading reconstructed points and the camera path…');
  try {
    const data = await api(`/api/runs/${id}/cloud`);
    if (selected !== id) return;
    if (!renderer) throw new Error('WebGL is unavailable. Try opening the app in Chrome or Edge.');
    if(gaussianViewer){await gaussianViewer.dispose();gaussianViewer=null;}dispose(cloud); dispose(pathGroup);dispose(surface);dispose(splats);dispose(inferredSurface);inferredSurface=null;surface=null;splats=null;
    const raw = new Float32Array(data.positions);
    // Focus the main cluster so a few remote outliers do not hide the useful scene.
    // All original points remain in the renderer and the download.
    const low=[], high=[];
    for(let axis=0;axis<3;axis++){
      const values=[];for(let i=axis;i<raw.length;i+=3)values.push(raw[i]);values.sort((a,b)=>a-b);
      low.push(values[Math.floor(values.length*.02)]);high.push(values[Math.min(values.length-1,Math.floor(values.length*.98))]);
    }
    const bounds = new THREE.Box3(new THREE.Vector3(...low),new THREE.Vector3(...high));
    const centre = bounds.getCenter(new THREE.Vector3());
    const extent = bounds.getSize(new THREE.Vector3());
    const scale = 16 / Math.max(extent.x, extent.y, extent.z, .001);
    // Ground-aligned exports already use +Y up. Legacy/raw COLMAP runs use
    // image-style +Y down and retain the old display flip.
    const ySign = data.ground_aligned ? 1 : -1;
    const pos = new Float32Array(raw.length);
    for (let i=0;i<raw.length;i+=3) {pos[i]=(raw[i]-centre.x)*scale;pos[i+1]=ySign*(raw[i+1]-centre.y)*scale;pos[i+2]=-(raw[i+2]-centre.z)*scale;}
    const colors = new Float32Array(data.colors.length);
    const color = new THREE.Color();
    for (let i=0;i<colors.length;i+=3) {color.setRGB(data.colors[i]/255,data.colors[i+1]/255,data.colors[i+2]/255,THREE.SRGBColorSpace);colors[i]=color.r;colors[i+1]=color.g;colors[i+2]=color.b;}
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(pos,3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors,3));
    cloud = new THREE.Points(geometry, new THREE.PointsMaterial({size:Number($('pointSize').value),sizeAttenuation:false,vertexColors:true}));
    cloud.visible = $('cloudToggle').checked; scene.add(cloud);
    pathGroup = new THREE.Group();
    const centres = data.cameras.map(c => new THREE.Vector3((c.centre[0]-centre.x)*scale,ySign*(c.centre[1]-centre.y)*scale,-(c.centre[2]-centre.z)*scale));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(centres),new THREE.LineBasicMaterial({color:0x70ebc5,transparent:true,opacity:.8}));
    pathGroup.add(line);
    const cameraPoints = new THREE.Points(new THREE.BufferGeometry().setFromPoints(centres), new THREE.PointsMaterial({color:0xafffe2,size:5,sizeAttenuation:false}));
    pathGroup.add(cameraPoints); pathGroup.visible = $('pathToggle').checked;scene.add(pathGroup);

    $('splatLayer').hidden=!data.gaussian_splats;
    $('completionLayer').hidden=!data.completion_available;
    grid.position.y = data.ground_aligned ? -centre.y*scale-.02 : -extent.y * scale / 2 - .5;
    axes.position.set(0,grid.position.y+.03,0);
    $('viewLabel').textContent=data.ground_aligned?'Y UP (GREEN) · X RED · Z BLUE · HEADING ARBITRARY':'PERSPECTIVE';
    // Fit to the scene rather than distant camera centres; no geographic up is assumed.
    home = {target:new THREE.Vector3(),position:new THREE.Vector3(10,8,17)};
    if(data.surface_available){
      const surfaceData=await api(`/api/runs/${id}/surface`);
      if(selected!==id)return;
      const vertices=new Float32Array(surfaceData.positions);
      for(let i=0;i<vertices.length;i+=3){vertices[i]=(vertices[i]-centre.x)*scale;vertices[i+1]=ySign*(vertices[i+1]-centre.y)*scale;vertices[i+2]=-(vertices[i+2]-centre.z)*scale;}
      const vertexColors=new Float32Array(surfaceData.colors.length);
      for(let i=0;i<vertexColors.length;i+=3){color.setRGB(surfaceData.colors[i]/255,surfaceData.colors[i+1]/255,surfaceData.colors[i+2]/255,THREE.SRGBColorSpace);vertexColors[i]=color.r;vertexColors[i+1]=color.g;vertexColors[i+2]=color.b;}
      const surfaceGeo=new THREE.BufferGeometry();surfaceGeo.setAttribute('position',new THREE.BufferAttribute(vertices,3));surfaceGeo.setAttribute('color',new THREE.BufferAttribute(vertexColors,3));surfaceGeo.setIndex(surfaceData.indices);surfaceGeo.computeVertexNormals();
      let texture=null;
      if(surfaceData.texture&&surfaceData.uv){surfaceGeo.setAttribute('uv',new THREE.BufferAttribute(new Float32Array(surfaceData.uv),2));texture=await new THREE.TextureLoader().loadAsync(`/api/runs/${id}/download/${surfaceData.texture}`);texture.colorSpace=THREE.SRGBColorSpace;}
      const videoMaterial=new THREE.MeshBasicMaterial({map:texture,vertexColors:!texture,side:THREE.DoubleSide});
      const litMaterial=new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,roughness:.82,metalness:0});
      surface=new THREE.Mesh(surfaceGeo,videoMaterial);surface.userData.videoMaterial=videoMaterial;surface.userData.litMaterial=litMaterial;scene.add(surface);
      if(Number.isInteger(surfaceData.inferred_face_start)){
        const split=surfaceData.inferred_face_start*3;
        const inferredGeo=surfaceGeo.clone();inferredGeo.setIndex(surfaceData.indices.slice(split));
        surfaceGeo.setIndex(surfaceData.indices.slice(0,split));
        inferredSurface=new THREE.Mesh(inferredGeo,videoMaterial.clone());
        inferredSurface.visible=$('completionToggle').checked;scene.add(inferredSurface);
      }
      surface.userData.originalColors=surfaceGeo.getAttribute('color').clone();
      if(surfaceData.semantic_colors){const sem=new Float32Array(surfaceData.semantic_colors.length);for(let i=0;i<sem.length;i+=3){color.setRGB(surfaceData.semantic_colors[i]/255,surfaceData.semantic_colors[i+1]/255,surfaceData.semantic_colors[i+2]/255,THREE.SRGBColorSpace);sem[i]=color.r;sem[i+1]=color.g;sem[i+2]=color.b;}surface.userData.semanticColors=new THREE.BufferAttribute(sem,3);}
      surface.visible=$('surfaceToggle').checked;
      $('cloudToggle').checked=false;cloud.visible=false;
      const surfaceBounds=new THREE.Box3();
      for(const index of new Set(surfaceData.indices))surfaceBounds.expandByPoint(new THREE.Vector3(vertices[index*3],vertices[index*3+1],vertices[index*3+2]));
      const surfaceCentre=surfaceBounds.getCenter(new THREE.Vector3());
      const size=surfaceBounds.getSize(new THREE.Vector3());
      const distance=Math.max(size.x,size.y,size.z);
      home={target:surfaceCentre,position:surfaceCentre.clone().add(new THREE.Vector3(.45,.85,.95).multiplyScalar(distance))};
    }
    if(data.gaussian_splats){
      gaussianViewer=new GaussianSplats3D.Viewer({selfDrivenMode:false,renderer,camera,threeScene:scene,useBuiltInControls:false,
        gpuAcceleratedSort:false,sharedMemoryForWorkers:false,enableSIMDInSort:false,integerBasedSort:true,halfPrecisionCovariancesOnGPU:true,dynamicScene:false,
        renderMode:GaussianSplats3D.RenderMode.Always,sceneRevealMode:GaussianSplats3D.SceneRevealMode.Instant,
        sphericalHarmonicsDegree:0,antialiased:true,showLoadingUI:false});
      const splatLoad=gaussianViewer.addSplatScene(`/api/runs/${id}/download/gaussians.ply`,{splatAlphaRemovalThreshold:5,
        position:[-centre.x*scale,-ySign*centre.y*scale,centre.z*scale],rotation:[0,0,0,1],scale:[scale,ySign*scale,-scale],showLoadingUI:false});
      // Some embedded browsers never signal completion from the optional splat-tree worker,
      // even though the sorted GPU scene is already ready and visible.
      splatLoad.catch(error=>console.warn('Gaussian background indexing did not finish:',error));
      await Promise.race([splatLoad,new Promise(resolve=>setTimeout(resolve,4000))]);
      if(selected!==id)return;$('surfaceToggle').checked=false;surface.visible=false;$('cloudToggle').checked=false;cloud.visible=false;
      gaussianViewer.splatMesh.visible=$('splatToggle').checked;
      const evidenceCamera=data.cameras[Math.floor(data.cameras.length/2)];
      if(evidenceCamera?.world_to_camera){
        const cameraToWorld=new THREE.Matrix4().set(...evidenceCamera.world_to_camera.flat()).invert();
        const elements=cameraToWorld.elements;
        const rawPosition=new THREE.Vector3(elements[12],elements[13],elements[14]);
        const viewPosition=new THREE.Vector3((rawPosition.x-centre.x)*scale,ySign*(rawPosition.y-centre.y)*scale,-(rawPosition.z-centre.z)*scale);
        const forward=new THREE.Vector3(elements[8]*scale,elements[9]*ySign*scale,-elements[10]*scale).normalize();
        const viewUp=new THREE.Vector3(-elements[4]*scale,-elements[5]*ySign*scale,elements[6]*scale).normalize();
        camera.up.copy(viewUp);home={position:viewPosition,up:viewUp,target:viewPosition.clone().add(forward.multiplyScalar(Math.max(extent.x,extent.y,extent.z)*scale*.6))};
      }
    }
    fitView();
    const activeRun=runs.find(run=>run.id===id);
    if(activeRun?.metrics?.solid_house_models!==undefined)$('colorMode').value='rgb';
    if(activeRun?.metrics?.field_pose_frames&&activeRun.metrics.solid_house_models===undefined&&home){camera.position.copy(home.target).add(new THREE.Vector3(0,home.position.distanceTo(home.target),.001));controls.target.copy(home.target);controls.update();}
    $('colorMode').dispatchEvent(new Event('change'));
    $('modelMessage').hidden = true;
    $('pointCount').textContent = `${number(data.displayed_points)} points`;
    loaded = id;
  } catch(error) { if (selected === id) message('Could not open the model', error.message); }
  finally { if (cloudLoading === id) cloudLoading = null; }
}
function renderRuns() {
  const signature=JSON.stringify([selected,runs]);
  if(signature===runListSignature)return;
  runListSignature=signature;
  $('runCount').textContent = runs.length;
  $('runList').replaceChildren();
  for (const run of runs) {
    const button = document.createElement('button'); button.className = `run ${run.id===selected?'active':''}`;
    const title = document.createElement('strong'); title.textContent = run.synthetic?'Synthetic test':run.name; button.title=title.textContent;
    const status = document.createElement('small'); status.textContent = run.metrics.calibration_suspect?'Calibration failed':`${run.status==='complete'?'✓ ':''}${run.synthetic?'Test scene · ':''}${run.status==='complete'?'Model ready':run.stage}`;
    button.append(title,status);button.onclick=()=>selectRun(run.id);$('runList').append(button);
  }
}
function renderDetails(run) {
  $('buildAI').disabled=run.status!=='complete'||runs.some(r=>r.status==='running');
  $('trainGsplat').disabled=run.status!=='complete'||runs.some(r=>r.status==='running');
  const signature=JSON.stringify(run);
  if(signature===detailsSignature){if(run.status==='complete')loadCloud(run.id);return;}
  detailsSignature=signature;
  $('projectTitle').textContent = run.synthetic ? 'Synthetic test scene' : run.name;
  const m=run.metrics;
  const ai=['da3-small','moge-2'].includes(m.engine);
  const open3d=m.engine==='open3d-tsdf';
  const gsplat=m.engine==='gsplat-pytorch';
  const completed=m.engine==='semantic-completion';
  const photogrammetric=m.engine==='colmap-mvs'||open3d||completed;
  const fused=!!m.fusion&&!open3d;
  const partial=run.status==='complete'&&m.registered_ratio<.8;
  const elapsed=run.elapsed_s?(run.elapsed_s<60?`${Math.round(run.elapsed_s)} sec`:`${Math.round(run.elapsed_s/60)} min`):'';
  $('projectSubtitle').textContent = `${m.calibration_suspect?'Unreliable geometry — calibration failed':partial?'Partial sparse model ready':run.stage} · ${run.synthetic?'Ideal synthetic scene':'Drone video'}${elapsed?' · '+elapsed+' processing':''}`;
  $('modelBadge').textContent=gsplat?'PYTORCH 3D GAUSSIANS':open3d?'OPEN3D TSDF':photogrammetric?'DENSE PHOTOGRAMMETRY':ai?'AI-INFERRED GEOMETRY':m.calibration_suspect?'UNRELIABLE GEOMETRY':partial?'PARTIAL RECONSTRUCTION':m.intrinsics_fixed?'EXPERIMENTAL CALIBRATION':'SPARSE RECONSTRUCTION';
  $('modelBadge').style.color=ai||partial||m.calibration_suspect||m.intrinsics_fixed?'var(--orange)':'';
  $('surfaceLayer').hidden=!(ai||photogrammetric||gsplat);$('splatLayer').hidden=!gsplat;
  $('outputType').textContent=open3d?'Open3D refined surface':photogrammetric?'Dense verified point cloud':fused?'Full-video fused surface':ai?'AI depth + dense cloud':'Sparse point cloud';
  $('geometryType').textContent=open3d?'Calibrated TSDF':photogrammetric?'Multi-view stereo':ai?'AI-inferred':'Estimated';
  $('noticeTitle').textContent=ai?'AI preview, not a measured map':'Scale is not calibrated';
  $('noticeText').textContent=ai?'This surface comes from the middle view’s predicted depth. The point cloud combines multiple views. Neither is validated for measurement.':'A GPS file or known distance is needed before measuring in metres. Dense surfaces are not included yet.';
  $('viewsLabel').textContent=fused?'FRAMES FUSED':ai?'PREDICTED VIEWS':'REGISTERED FRAMES';
  $('pointsExplanation').textContent=ai?'Confidence-filtered AI predictions':'Triangulated visual features';
  $('qualityLabel').textContent=ai?'DEPTH MODEL':'REPROJECTION ERROR';
  $('qualityExplanation').textContent=m.engine==='moge-2'?'MoGe-2 + semantic sky exclusion':ai?'Local inference · Apache 2.0':'Image fit, not ground accuracy';
  $('registered').textContent = m.registered_images!==undefined?`${m.registered_images} / ${m.selected_images}`:'—';
  $('registrationRatio').textContent = m.registered_ratio!==undefined?`${(m.registered_ratio*100).toFixed(0)}% of selected frames`:'Awaiting reconstruction';
  $('points').textContent = m.points!==undefined?number(m.points):'—';
  $('reprojection').textContent = m.mean_reprojection_error_px!==undefined?`${m.mean_reprojection_error_px.toFixed(2)} px`:'—';
  if(ai){$('registered').textContent=m.predicted_views||'—';$('registrationRatio').textContent='Views jointly predicted';$('reprojection').textContent='DA3 Small';}
  if(photogrammetric){$('viewsLabel').textContent='REGISTERED KEYFRAMES';$('pointsExplanation').textContent='Geometrically verified multi-view points';$('qualityLabel').textContent='IMAGE FIT';$('qualityExplanation').textContent='COLMAP reprojection error';$('noticeTitle').textContent='Observed geometry · relative scale';$('noticeText').textContent='Only surfaces supported by overlapping calibrated views are shown. Missing regions are left incomplete. GPS, GCPs, or a known distance are still needed for metric scale.';}
  if(open3d){$('pointsExplanation').textContent='TSDF surface vertices';$('qualityLabel').textContent='DEPTH SAMPLES';$('reprojection').textContent=number(m.valid_depth_pixels);$('qualityExplanation').textContent='Calibrated non-sky pixels fused';$('noticeTitle').textContent='Open3D fused geometry · relative scale';$('noticeText').textContent='Open3D TSDF combines COLMAP geometric depth from all registered views, removes small fragments, and smooths stereo noise. Unsupported areas remain incomplete. GPS, GCPs, or a known distance are still needed for metric scale.';}
  if(completed){$('modelBadge').textContent='OBSERVED + INFERRED COMPLETION';$('outputType').textContent='Textured mesh + inferred backs';$('geometryType').textContent='Stereo + semantic ground fit';$('qualityLabel').textContent='CLOSED OBJECT CLUSTERS';$('reprojection').textContent=`${m.closed_building_clusters} buildings · ${m.closed_vegetation_clusters} vegetation`;$('qualityExplanation').textContent='Approximate closed backs, not measured objects';$('noticeTitle').textContent='Completion is an approximation';$('noticeText').textContent='Sky and unsupported vertices are rejected. Ground is refitted from stereo-supported semantic ground. Inferred backs use colours from the same object; toggle completion off to inspect only observed geometry.';}
  if(gsplat){$('outputType').textContent='Open3D mesh + trained 3D Gaussians';$('geometryType').textContent='MVS / TSDF calibrated';$('viewsLabel').textContent='TRAIN / HOLDOUT VIEWS';$('registered').textContent=`${m.training_views} / ${m.holdout_views}`;$('registrationRatio').textContent=`${m.training_steps.toLocaleString()} PyTorch steps`;$('pointsExplanation').textContent='Optimized Gaussian primitives';$('qualityLabel').textContent='HELD-OUT PSNR';$('reprojection').textContent=`${m.holdout_psnr_db.toFixed(2)} dB`;$('qualityExplanation').textContent='Video-versus-render image fit';$('noticeTitle').textContent='Gaussian appearance · geographic alignment pending';$('noticeText').textContent='The calibrated cameras and Open3D mesh initialize the scene; gsplat optimizes appearance against non-sky video pixels. Fill the telemetry CSV with real per-frame GPS/IMU to produce metre-scale WGS84 local coordinates.';}
  if(fused){$('registered').textContent=`${m.frames_integrated||0} / ${m.frames_decoded||'…'}`;$('registrationRatio').textContent=m.coverage_percent?`${m.coverage_percent.toFixed(0)}% of video · ${m.last_timestamp_s.toFixed(2)} seconds`:'Processing every video frame';$('noticeTitle').textContent='Full-video fusion · relative scale';$('noticeText').textContent='Depth from all integrated frames contributes to one shared surface. Geometry and camera positions remain unvalidated; gaps may remain where the video provides insufficient evidence.';}
  if(m.engine==='moge-2'){$('reprojection').textContent='MoGe-2 Base';$('noticeTitle').textContent='Estimated ground plane · Y up';$('noticeText').textContent='Sky is excluded by semantic labels. A rigid rotation aligns estimated ground with X/Z, preserving height. Semantic colors: green ground, orange buildings, dark green vegetation, gray other. Coordinates use relative scale; dimensions are not metres.';}
  if(m.aerial_corrector_frames!==undefined){$('qualityLabel').textContent='TRAINED CORRECTOR';$('reprojection').textContent=`${m.aerial_corrector_frames} / ${m.frames_integrated} frames`;$('qualityExplanation').textContent='Accepted only when held-out feature geometry improved';}
  if(m.tracked_frames!==undefined){$('qualityLabel').textContent='CAMERA POSES';$('reprojection').textContent=`${m.tracked_frames} tracked + ${m.registered_frames} anchors`;$('qualityExplanation').textContent=`${m.fallback_frames} interpolated · depth corrector accepted on ${m.aerial_corrector_frames} frames`;}
  if(m.field_pose_frames!==undefined){$('qualityLabel').textContent='FLAT FIELD';$('reprojection').textContent=`${m.field_pose_frames} / ${m.frames_integrated} levelled`;$('qualityExplanation').textContent=`${(m.field_plane_support_median*100).toFixed(0)}% median plane support · ${m.tracked_frames} camera poses tracked`;}
  if(m.field_pose_frames!==undefined){$('noticeTitle').textContent='Flat-field constraint · Y up';$('noticeText').textContent='Semantic ground is projected onto one shared field plane. Houses and vegetation retain bounded relief below the drone altitude. The camera path is hidden by default. Scale and building dimensions remain unvalidated.';}
  if(m.solid_house_models!==undefined){$('qualityLabel').textContent=m.texture_atlas?'PHOTO TEXTURE':'OBJECT MODELS';$('reprojection').textContent=m.texture_atlas?`${m.texture_atlas.resolution} × ${m.texture_atlas.resolution} atlas`:`${m.solid_house_models} houses + ${m.solid_tree_models} trees`;$('qualityExplanation').textContent=m.texture_atlas?`${number(m.texture_atlas.samples)} aligned video samples`:`Closed solids placed from ${m.frames_integrated} tracked video frames`;$('noticeTitle').textContent=m.texture_atlas?'Video texture atlas · approximate geometry':'Video-coloured objects · approximate dimensions';$('noticeText').textContent=m.texture_atlas?'A photographic texture is baked from aligned frames and applied to the 3D surface. Upward-facing visible areas carry the strongest detail. Vertical and hidden faces remain inferred because the flight does not observe every side.':'Visible house and tree colours are transferred from matching video observations. Generated walls, roofs, trunks, and hidden sides remain approximate because the video has no GPS, calibration, or complete side views. Use the colour menu to inspect semantic classes.';}
  const ready=run.status==='complete';
  $('downloadCloud').classList.toggle('disabled',!ready);$('downloadCloud').setAttribute('aria-disabled',String(!ready));
  $('downloadCloud').href=`/api/runs/${run.id}/download/${gsplat?'gaussians':ai||photogrammetric?'dense':'sparse'}.ply`;
  $('exports').replaceChildren();
  for (const [file,label] of [['camera_centres.csv','Camera positions'],['REPORT.md','Reconstruction report'],['metrics.json','Quality statistics']]) {
    const link=document.createElement('a');link.textContent=label;const arrow=document.createElement('span');arrow.textContent='↓';link.append(arrow);
    link.href=`/api/runs/${run.id}/download/${file}`;if(!ready)link.className='disabled';$('exports').append(link);
  }
  if(ai){const link=document.createElement('a');link.href=`/api/runs/${run.id}/download/surface.glb`;link.textContent=fused?'Fused surface (GLB) ↓':'AI depth surface (GLB) ↓';$('exports').append(link);}
  if(photogrammetric){const link=document.createElement('a');link.href=`/api/runs/${run.id}/download/surface.glb`;link.textContent=open3d?'Open3D surface (GLB) ↓':'Calibrated textured mesh (GLB) ↓';$('exports').append(link);}
  if(gsplat){for(const [file,label] of [['gaussians.ply','3D Gaussian Splat (PLY)'],['gaussians.pt','PyTorch checkpoint'],['telemetry_template.csv','GPS + IMU template'],['gsplat_metrics.json','Gaussian validation']]){const link=document.createElement('a');link.href=`/api/runs/${run.id}/download/${file}`;link.textContent=`${label} ↓`;$('exports').append(link);}}
  if(fused){const link=document.createElement('a');link.href=`/api/runs/${run.id}/download/frames.csv`;link.textContent='Every-frame coverage (CSV) ↓';$('exports').append(link);}
  const pipelineProgress=run.progress||{};
  const stages=fused?['Decode every frame','Shared camera alignment','Multi-view fusion','Combined surface']:ai?['Video check','AI depth','Backprojection','Dense preview']:(pipelineProgress.stages||['Video analysis','Keyframe selection','Feature extraction','Frame matching','Sparse reconstruction','Quality report']);
  let current=Number.isInteger(pipelineProgress.stage_index)?pipelineProgress.stage_index:(run.stage.includes('features')?1:run.stage.includes('Matching')?2:run.stage.includes('3D')?3:0);
  if(fused&&m.frames_integrated)current=m.frames_integrated===m.frames_decoded?3:2;
  if(ready)current=4;
  $('stages').replaceChildren();
  stages.forEach((label,i)=>{const stage=document.createElement('div');stage.className=`stage ${i<current||ready?'done':i===current?'current':''}`;const icon=document.createElement('b');icon.textContent=i<current||ready?'✓':i+1;stage.append(icon,document.createTextNode(label));if(i===current&&!ready&&pipelineProgress.percent!==undefined){const pct=document.createElement('small');pct.textContent=` ${pipelineProgress.percent.toFixed(0)}%`;stage.append(pct);}$('stages').append(stage);});
  $('errorBox').hidden=!(run.error||run.warnings.length);
  $('errorBox').textContent=run.error||run.warnings.join(' ');
  if(ready)loadCloud(run.id);
  else message(run.status==='failed'?'Reconstruction needs attention':run.stage,run.error||'Processing your video locally. The 3D result will appear here when it is ready. You can inspect the source video and keyframes now.');
}
async function selectRun(id) {
  selected=id; framesLoaded=null;loaded=null;
  detailsSignature='';
  if(gaussianViewer){await gaussianViewer.dispose();gaussianViewer=null;}dispose(cloud);dispose(pathGroup);dispose(surface);dispose(splats);dispose(inferredSurface);inferredSurface=null;cloud=null;pathGroup=null;surface=null;splats=null;
  $('cloudToggle').checked=true;$('surfaceToggle').checked=true;$('splatToggle').checked=true;$('pathToggle').checked=false;
  $('completionToggle').checked=true;
  const url=new URL(location.href);url.searchParams.set('result',id);history.replaceState(null,'',url);
  $('pointCount').textContent='— points';
  $('sourceVideo').src=`/api/runs/${id}/video`;
  renderRuns();
  const run=runs.find(r=>r.id===id);if(run)renderDetails(run);
  if(currentView==='frames')loadFrames();
  if(!$('logs').hidden)loadLogs();
}
async function loadFrames() {
  if(!selected||framesLoaded===selected)return;
  const id=selected;
  try {
    const frames=await api(`/api/runs/${id}/frames`);if(selected!==id)return;
    $('framesView').replaceChildren();
    if(!frames.length){$('framesView').textContent='Keyframes will appear after video preparation.';return;}
    frames.forEach(frame=>{const figure=document.createElement('figure');const img=document.createElement('img');img.src=frame.url;img.alt=frame.name;img.loading='lazy';const caption=document.createElement('figcaption');caption.textContent=frame.name;figure.append(img,caption);$('framesView').append(figure);});
    if(runs.find(r=>r.id===id)?.status==='complete')framesLoaded=id;
  }catch(error){notify(error.message);}
}
async function loadLogs(){if(!selected)return;const id=selected;try{const result=await api(`/api/runs/${id}/logs`);if(selected===id){$('logs').textContent=result.text||'Waiting for worker output…';$('logs').scrollTop=$('logs').scrollHeight;}}catch(error){notify(error.message);}}
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{
  currentView=button.dataset.view;
  document.querySelectorAll('[data-view]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-selected',String(b===button));});
  $('sourceView').hidden=currentView!=='source';$('framesView').hidden=currentView!=='frames';
  for(const id of ['viewTools','viewFooter','viewLabel'])$(id).hidden=currentView!=='model';
  if(currentView!=='source')$('sourceVideo').pause();
  if(currentView==='frames')loadFrames();
});
$('resetView').onclick=fitView;
$('topView').onclick=()=>{if(home){camera.position.copy(home.target).add(new THREE.Vector3(0,home.position.distanceTo(home.target),.001));controls.target.copy(home.target);controls.update();}};
$('rotateView').onclick=()=>{if(controls){controls.autoRotate=!controls.autoRotate;$('rotateView').setAttribute('aria-pressed',String(controls.autoRotate));}};
$('cloudToggle').onchange=e=>{if(cloud)cloud.visible=e.target.checked;};
$('surfaceToggle').onchange=e=>{if(surface)surface.visible=e.target.checked;};
$('completionToggle').onchange=e=>{if(inferredSurface)inferredSurface.visible=e.target.checked;};
$('splatToggle').onchange=e=>{if(gaussianViewer?.splatMesh)gaussianViewer.splatMesh.visible=e.target.checked;if(splats)splats.visible=e.target.checked;};
$('pathToggle').onchange=e=>{if(pathGroup)pathGroup.visible=e.target.checked;};
$('gridToggle').onchange=e=>{if(grid)grid.visible=e.target.checked;};
$('pointSize').oninput=e=>{$('pointSizeValue').value=e.target.value;if(cloud)cloud.material.size=Number(e.target.value);};
$('colorMode').onchange=e=>{if(surface){surface.geometry.setAttribute('color',e.target.value==='semantic'&&surface.userData.semanticColors?surface.userData.semanticColors:surface.userData.originalColors);surface.material=e.target.value==='rgb'?surface.userData.videoMaterial:surface.userData.litMaterial;surface.material.needsUpdate=true;}if(cloud){cloud.material.vertexColors=e.target.value==='rgb';cloud.material.color.set(e.target.value==='rgb'?0xffffff:0x70ebc5);cloud.material.needsUpdate=true;}};
$('toggleLogs').onclick=()=>{$('logs').hidden=!$('logs').hidden;$('toggleLogs').textContent=$('logs').hidden?'Show log ↓':'Hide log ↑';$('toggleLogs').setAttribute('aria-expanded',String(!$('logs').hidden));if(!$('logs').hidden)loadLogs();};
$('newRun').onclick=()=>$('uploadDialog').showModal();$('closeDialog').onclick=()=>$('uploadDialog').close();
$('buildAI').onclick=async()=>{
  if(!selected)return;$('buildAI').disabled=true;
  try{const result=await api(`/api/runs/${selected}/ai`,{method:'POST',headers:{'X-AeroRecon':'local'}});notify('AI reconstruction is starting on your GPU.');setTimeout(async()=>{await refresh();if(runs.some(r=>r.id===result.id))selectRun(result.id);},1500);}
  catch(error){notify(error.message);$('buildAI').disabled=false;}
};
$('trainGsplat').onclick=async()=>{
  if(!selected)return;$('trainGsplat').disabled=true;
  try{const result=await api(`/api/runs/${selected}/gsplat`,{method:'POST',headers:{'X-AeroRecon':'local'}});notify('Gaussian training is starting on your GPU.');setTimeout(async()=>{await refresh();if(runs.some(r=>r.id===result.id))selectRun(result.id);},1500);}
  catch(error){notify(error.message);$('trainGsplat').disabled=false;}
};
$('videoFile').onchange=()=>{$('fileName').textContent=$('videoFile').files[0]?.name||'Choose or drop a video';};
for(const event of ['dragenter','dragover'])$('dropzone').addEventListener(event,()=> $('dropzone').classList.add('dragging'));
for(const event of ['dragleave','drop'])$('dropzone').addEventListener(event,()=> $('dropzone').classList.remove('dragging'));
$('uploadForm').onsubmit=async e=>{
  e.preventDefault();const file=$('videoFile').files[0];if(!file)return;
  if(file.size>1024**3){$('uploadError').textContent='Choose a video smaller than 1 GB.';return;}
  $('submitUpload').disabled=true;$('closeDialog').disabled=true;$('uploadError').textContent='';$('submitUpload').textContent='Uploading video…';
  try {const form=new FormData();form.append('file',file);const result=await api('/api/runs',{method:'POST',headers:{'X-AeroRecon':'local'},body:form});
    $('uploadDialog').close();notify(result.reused?'Identical video found. Opening its existing reconstruction; use Build full-video map to rebuild.':'Video received. Reconstruction is starting.');
    // The worker creates the run directory asynchronously.
    setTimeout(async()=>{await refresh();if(runs.some(r=>r.id===result.id))selectRun(result.id);},1500);
  }catch(error){$('uploadError').textContent=error.message;}
  finally{$('submitUpload').disabled=false;$('closeDialog').disabled=false;$('submitUpload').textContent='Build 3D reconstruction →';}
};
async function refresh(){
  try{runs=await api('/api/runs');renderRuns();if(!selected&&runs.length){const requested=new URLSearchParams(location.search).get('result');await selectRun(runs.some(r=>r.id===requested)?requested:runs[0].id);}else{const run=runs.find(r=>r.id===selected);if(run)renderDetails(run);}
    if(!runs.length)message('Your next map starts here','Choose New reconstruction to upload your drone video.');
    if(!$('logs').hidden)loadLogs();if(currentView==='frames')loadFrames();
  }catch(error){message('Cannot reach the local app','The processing server may have stopped. Start AeroRecon again using Start-AeroRecon.cmd.');}
}
try{setupScene();}catch(error){notify('3D graphics are unavailable in this browser. Video and downloads still work.');}
await refresh();setInterval(refresh,4000);
