import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Layers, 
  Grid3X3, 
  ArrowRight, 
  Cpu, 
  Image as ImageIcon, 
  Hash, 
  Maximize2,
  ChevronRight,
  ChevronLeft,
  Info
} from 'lucide-react';
import * as d3 from 'd3';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Constants ---
const PATCH_SIZE = 4; // 4x4 patches for visualization
const GRID_SIZE = 4;  // 4x4 grid of patches (16 total)
const EMBED_DIM = 64;
const IMAGE_URL = "https://picsum.photos/seed/vit/400/400";

// --- Types ---
type Stage = 'input' | 'patching' | 'projection' | 'positional' | 'encoder';

interface StageInfo {
  id: Stage;
  title: string;
  description: string;
  icon: React.ReactNode;
}

const STAGES: StageInfo[] = [
  { 
    id: 'input', 
    title: 'Input Image', 
    description: 'The process starts with a standard 2D image of size H x W x C.',
    icon: <ImageIcon size={20} />
  },
  { 
    id: 'patching', 
    title: 'Patch Partition', 
    description: 'The image is divided into fixed-size patches (P x P). Each patch is treated like a "word" in a sentence.',
    icon: <Grid3X3 size={20} />
  },
  { 
    id: 'projection', 
    title: 'Linear Projection', 
    description: 'Each patch is flattened and projected into a D-dimensional embedding space.',
    icon: <Maximize2 size={20} />
  },
  { 
    id: 'positional', 
    title: 'Positional Encoding', 
    description: 'Since Transformers have no inherent sense of order, we add learnable position embeddings to the patch tokens.',
    icon: <Hash size={20} />
  },
  { 
    id: 'encoder', 
    title: 'Transformer Encoder', 
    description: 'The sequence of tokens passes through multiple layers of Multi-Head Self-Attention and MLP blocks.',
    icon: <Cpu size={20} />
  }
];

// --- Components ---

const PatchGrid = ({ stage }: { stage: Stage }) => {
  const patches = Array.from({ length: GRID_SIZE * GRID_SIZE });
  
  return (
    <div className="relative w-64 h-64 bg-zinc-900 rounded-lg overflow-hidden border border-zinc-800 shadow-2xl">
      <img 
        src={IMAGE_URL} 
        alt="Input" 
        className={cn(
          "absolute inset-0 w-full h-full object-cover transition-opacity duration-500",
          stage === 'input' ? "opacity-100" : "opacity-40"
        )}
        referrerPolicy="no-referrer"
      />
      
      <div className="absolute inset-0 grid grid-cols-4 grid-rows-4">
        {patches.map((_, i) => (
          <motion.div
            key={i}
            initial={false}
            animate={{
              borderWidth: stage !== 'input' ? 1 : 0,
              borderColor: 'rgba(255, 255, 255, 0.2)',
              scale: stage === 'patching' ? 0.9 : 1,
            }}
            className="w-full h-full flex items-center justify-center"
          >
            {stage === 'patching' && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-[10px] text-zinc-400 font-mono"
              >
                {i}
              </motion.div>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
};

const ProjectionView = ({ stage }: { stage: Stage }) => {
  const patches = Array.from({ length: GRID_SIZE * GRID_SIZE });
  
  return (
    <div className="flex flex-col items-center gap-8 w-full max-w-2xl">
      <div className="flex flex-wrap justify-center gap-2">
        {patches.map((_, i) => (
          <motion.div
            key={i}
            layoutId={`patch-${i}`}
            className="w-12 h-12 bg-zinc-800 rounded border border-zinc-700 flex items-center justify-center overflow-hidden"
            animate={{
              y: stage === 'projection' ? [0, -20, 0] : 0,
              transition: { delay: i * 0.05 }
            }}
          >
            <div className="w-full h-full bg-gradient-to-br from-indigo-500/20 to-purple-500/20" />
          </motion.div>
        ))}
      </div>
      
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col items-center gap-4"
      >
        <ArrowRight className="rotate-90 text-zinc-500" />
        <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-xl shadow-xl w-full text-center">
          <span className="text-sm font-mono text-indigo-400">Linear Projection (W_p)</span>
          <div className="mt-2 h-1 w-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full" />
        </div>
        <ArrowRight className="rotate-90 text-zinc-500" />
      </motion.div>

      <div className="flex flex-wrap justify-center gap-2">
        {patches.map((_, i) => (
          <motion.div
            key={`token-${i}`}
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 + i * 0.05 }}
            className="w-4 h-24 bg-indigo-500/20 border border-indigo-500/40 rounded-full flex flex-col items-center justify-end p-1 gap-1"
          >
            <div className="w-full h-1/3 bg-indigo-400/40 rounded-full" />
            <div className="w-full h-1/4 bg-purple-400/40 rounded-full" />
            <div className="w-full h-1/2 bg-indigo-500/60 rounded-full" />
          </motion.div>
        ))}
      </div>
    </div>
  );
};

const PositionalEncodingView = () => {
  const patches = Array.from({ length: GRID_SIZE * GRID_SIZE });
  
  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-2xl">
      <div className="grid grid-cols-8 gap-4">
        {patches.map((_, i) => (
          <div key={i} className="flex flex-col items-center gap-2">
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: i * 0.05 }}
              className="w-8 h-20 bg-indigo-500/20 border border-indigo-500/40 rounded-lg relative overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-t from-indigo-500/20 to-transparent" />
            </motion.div>
            <div className="text-zinc-500 text-[10px] font-mono">E_{i}</div>
            <div className="text-zinc-400 font-bold">+</div>
            <motion.div
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.5 + i * 0.05 }}
              className="w-8 h-8 bg-emerald-500/20 border border-emerald-500/40 rounded-lg flex items-center justify-center"
            >
              <Hash size={12} className="text-emerald-400" />
            </motion.div>
            <div className="text-zinc-500 text-[10px] font-mono">P_{i}</div>
          </div>
        ))}
      </div>
      
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
        className="mt-8 p-4 bg-zinc-900/50 border border-zinc-800 rounded-lg text-xs text-zinc-400 italic"
      >
        "Tokens now contain both visual content (E) and spatial context (P)."
      </motion.div>
    </div>
  );
};

const EncoderArchitecture = () => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    
    const width = 600;
    const height = 400;
    const svg = d3.select(containerRef.current)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${width} ${height}`);

    // Draw the main block
    const block = svg.append('g').attr('transform', 'translate(100, 50)');
    
    // Layer Norm 1
    block.append('rect')
      .attr('width', 400)
      .attr('height', 40)
      .attr('rx', 8)
      .attr('fill', '#18181b')
      .attr('stroke', '#3f3f46');
    block.append('text')
      .attr('x', 200)
      .attr('y', 25)
      .attr('text-anchor', 'middle')
      .attr('fill', '#a1a1aa')
      .attr('font-size', '12px')
      .attr('font-family', 'monospace')
      .text('Layer Norm');

    // Multi-Head Attention
    block.append('rect')
      .attr('x', 50)
      .attr('y', 70)
      .attr('width', 300)
      .attr('height', 80)
      .attr('rx', 12)
      .attr('fill', 'rgba(99, 102, 241, 0.1)')
      .attr('stroke', 'rgba(99, 102, 241, 0.5)');
    block.append('text')
      .attr('x', 200)
      .attr('y', 115)
      .attr('text-anchor', 'middle')
      .attr('fill', '#818cf8')
      .attr('font-weight', 'bold')
      .text('Multi-Head Self-Attention');

    // Layer Norm 2
    block.append('rect')
      .attr('y', 180)
      .attr('width', 400)
      .attr('height', 40)
      .attr('rx', 8)
      .attr('fill', '#18181b')
      .attr('stroke', '#3f3f46');
    block.append('text')
      .attr('x', 200)
      .attr('y', 205)
      .attr('text-anchor', 'middle')
      .attr('fill', '#a1a1aa')
      .attr('font-size', '12px')
      .attr('font-family', 'monospace')
      .text('Layer Norm');

    // MLP
    block.append('rect')
      .attr('x', 50)
      .attr('y', 250)
      .attr('width', 300)
      .attr('height', 80)
      .attr('rx', 12)
      .attr('fill', 'rgba(168, 85, 247, 0.1)')
      .attr('stroke', 'rgba(168, 85, 247, 0.5)');
    block.append('text')
      .attr('x', 200)
      .attr('y', 295)
      .attr('text-anchor', 'middle')
      .attr('fill', '#c084fc')
      .attr('font-weight', 'bold')
      .text('MLP (Feed Forward)');

    // Residual Connections
    const lineGenerator = d3.line();
    
    // Residual 1
    svg.append('path')
      .attr('d', lineGenerator([[80, 70], [80, 180], [150, 180]]))
      .attr('fill', 'none')
      .attr('stroke', '#52525b')
      .attr('stroke-dasharray', '4');

    return () => {
      d3.select(containerRef.current).selectAll('*').remove();
    };
  }, []);

  return (
    <div className="w-full h-[450px] bg-zinc-950/50 rounded-2xl border border-zinc-800 p-4 relative overflow-hidden">
      <div ref={containerRef} className="w-full h-full" />
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        <div className="flex items-center gap-2 text-[10px] text-zinc-500">
          <div className="w-3 h-3 bg-indigo-500/20 border border-indigo-500/50 rounded" />
          Attention Block
        </div>
        <div className="flex items-center gap-2 text-[10px] text-zinc-500">
          <div className="w-3 h-3 bg-purple-500/20 border border-purple-500/50 rounded" />
          MLP Block
        </div>
      </div>
    </div>
  );
};

export default function App() {
  const [currentStage, setCurrentStage] = useState<number>(0);
  const stage = STAGES[currentStage];

  const nextStage = () => setCurrentStage(prev => Math.min(prev + 1, STAGES.length - 1));
  const prevStage = () => setCurrentStage(prev => Math.max(prev - 1, 0));

  return (
    <div className="min-h-screen bg-[#050505] text-zinc-100 font-sans selection:bg-indigo-500/30">
      {/* Header */}
      <header className="border-b border-zinc-800/50 bg-black/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Cpu size={18} className="text-white" />
            </div>
            <h1 className="font-bold tracking-tight text-lg">ViT Visualizer</h1>
          </div>
          <div className="flex items-center gap-4">
            <a 
              href="https://arxiv.org/abs/2010.11929" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors flex items-center gap-1"
            >
              <Info size={14} />
              Original Paper
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-12 grid grid-cols-1 lg:grid-cols-12 gap-12">
        {/* Left Column: Navigation & Info */}
        <div className="lg:col-span-4 space-y-8">
          <div className="space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-[10px] font-bold uppercase tracking-wider">
              Step {currentStage + 1} of {STAGES.length}
            </div>
            <h2 className="text-4xl font-bold tracking-tight leading-tight">
              {stage.title}
            </h2>
            <p className="text-zinc-400 leading-relaxed">
              {stage.description}
            </p>
          </div>

          {/* Progress Steps */}
          <nav className="space-y-2">
            {STAGES.map((s, idx) => (
              <button
                key={s.id}
                onClick={() => setCurrentStage(idx)}
                className={cn(
                  "w-full flex items-center gap-4 p-4 rounded-xl transition-all duration-300 text-left group",
                  currentStage === idx 
                    ? "bg-zinc-900 border border-zinc-800 shadow-xl" 
                    : "hover:bg-zinc-900/50 text-zinc-500"
                )}
              >
                <div className={cn(
                  "w-10 h-10 rounded-lg flex items-center justify-center transition-colors",
                  currentStage === idx ? "bg-indigo-600 text-white" : "bg-zinc-800 group-hover:bg-zinc-700"
                )}>
                  {s.icon}
                </div>
                <div className="flex-1">
                  <div className={cn(
                    "text-sm font-semibold",
                    currentStage === idx ? "text-zinc-100" : "text-zinc-400"
                  )}>
                    {s.title}
                  </div>
                  <div className="text-[10px] opacity-60 font-mono">
                    {idx === 0 ? "INPUT" : idx === STAGES.length - 1 ? "OUTPUT" : "TRANSFORM"}
                  </div>
                </div>
                {currentStage === idx && (
                  <motion.div layoutId="active-indicator">
                    <ChevronRight size={16} className="text-indigo-400" />
                  </motion.div>
                )}
              </button>
            ))}
          </nav>

          {/* Controls */}
          <div className="flex items-center gap-4 pt-4">
            <button
              onClick={prevStage}
              disabled={currentStage === 0}
              className="flex-1 h-12 rounded-xl border border-zinc-800 flex items-center justify-center gap-2 hover:bg-zinc-900 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
            >
              <ChevronLeft size={20} />
              Back
            </button>
            <button
              onClick={nextStage}
              disabled={currentStage === STAGES.length - 1}
              className="flex-[2] h-12 rounded-xl bg-indigo-600 text-white font-bold flex items-center justify-center gap-2 hover:bg-indigo-500 disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-lg shadow-indigo-600/20"
            >
              Next Step
              <ChevronRight size={20} />
            </button>
          </div>
        </div>

        {/* Right Column: Visualization Stage */}
        <div className="lg:col-span-8 min-h-[600px] flex items-center justify-center bg-zinc-900/20 rounded-3xl border border-zinc-800/50 p-8 relative overflow-hidden">
          {/* Background Grid Accent */}
          <div className="absolute inset-0 opacity-[0.03] pointer-events-none" 
               style={{ backgroundImage: 'radial-gradient(#fff 1px, transparent 1px)', backgroundSize: '32px 32px' }} />
          
          <AnimatePresence mode="wait">
            <motion.div
              key={stage.id}
              initial={{ opacity: 0, scale: 0.95, filter: 'blur(10px)' }}
              animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
              exit={{ opacity: 0, scale: 1.05, filter: 'blur(10px)' }}
              transition={{ duration: 0.5, ease: [0.23, 1, 0.32, 1] }}
              className="w-full flex justify-center"
            >
              {stage.id === 'input' && <PatchGrid stage="input" />}
              {stage.id === 'patching' && <PatchGrid stage="patching" />}
              {stage.id === 'projection' && <ProjectionView stage="projection" />}
              {stage.id === 'positional' && <PositionalEncodingView />}
              {stage.id === 'encoder' && <EncoderArchitecture />}
            </motion.div>
          </AnimatePresence>

          {/* Stage Label Overlay */}
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 px-4 py-2 rounded-full bg-black/40 backdrop-blur-sm border border-white/5 text-[10px] font-mono text-zinc-500 uppercase tracking-widest">
            Interactive Visualization Stage
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-6 py-12 border-t border-zinc-800/50 mt-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="space-y-4">
            <div className="text-sm font-bold">About ViT</div>
            <p className="text-xs text-zinc-500 leading-relaxed">
              The Vision Transformer (ViT) applies the Transformer architecture directly to sequences of image patches. 
              It proved that convolution is not strictly necessary for competitive computer vision results.
            </p>
          </div>
          <div className="space-y-4">
            <div className="text-sm font-bold">Key Components</div>
            <ul className="text-xs text-zinc-500 space-y-2">
              <li>• Patch + Position Embeddings</li>
              <li>• Transformer Encoder Blocks</li>
              <li>• Multi-Head Self-Attention</li>
              <li>• MLP Head for Classification</li>
            </ul>
          </div>
          <div className="space-y-4">
            <div className="text-sm font-bold">Architecture Details</div>
            <p className="text-xs text-zinc-500 leading-relaxed">
              Input: (224, 224, 3) <br/>
              Patches: (16, 16) <br/>
              Sequence Length: 196 + 1 (CLS) <br/>
              Embedding Dim: 768 (ViT-Base)
            </p>
          </div>
        </div>
        <div className="mt-12 pt-8 border-t border-zinc-800/30 text-center text-[10px] text-zinc-600">
          Built with React, D3.js, and Framer Motion • 2026
        </div>
      </footer>
    </div>
  );
}
