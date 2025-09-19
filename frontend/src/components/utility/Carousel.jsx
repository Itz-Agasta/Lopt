import { motion, useAnimation, useMotionValue } from "framer-motion";
import { useState, useEffect, useRef } from "react";
import image1 from "../../assets/images/ScarlettJohansson.jpg";
import image2 from "../../assets/images/RishiSunak1.jpg";
import image3 from "../../assets/images/RashmikaMadanna.jpg";
import image4 from "../../assets/images/TaylorSwift.jpg";
import image5 from "../../assets/images/KristenBell.jpg";
import image6 from "../../assets/images/JohnnyDepp.jpg";
import image7 from "../../assets/images/MeganFox.jpg";
import image8 from "../../assets/images/KeanuReeves.jpg";

const slides = [
  {
    content: image1,
    name: "Scarlett Johansson (Deepfake)",
  },
  {
    content: image2,
    name: "Rishi Sunak (REAL)",
  },
  {
    content: image3,
    name: "Rashmika Madanna (Deepfake)",
  },
  {
    content: image4,
    name: "Taylor Swift (Deepfake)",
  },
  {
    content: image5,
    name: "Kristen Bell (Deepfake)",
  },
  {
    content: image6,
    name: "Johnny Depp (Deepfake)",
  },
  {
    content: image7,
    name: "Megan Fox (REAL)",
  },
  {
    content: image8,
    name: "Keanue Reeves (Deepfake)",
  },
];
const slideWidth = 320; // Increased from 270

export function InfiniteCarousel() {
  const controls = useAnimation();
  const [isAnimating, setIsAnimating] = useState(true);

  useEffect(() => {
    const animate = async () => {
      if (isAnimating) {
        await controls.start({
          x: -slides.length * slideWidth,
          transition: { 
            duration: 20, 
            ease: "linear",
            repeat: Infinity
          },
        });
      }
    };
    animate();
  }, [controls, isAnimating]);

  const duplicatedSlides = [...slides, ...slides, ...slides]; // Triple for smoother loop

  return (
    <div
      style={{
        overflow: "hidden",
        width: "100%",
        margin: "2rem auto",
        display: "flex",
        flexDirection: "row",
        flexShrink: 0,
        userSelect: "none", // Prevent text selection
        WebkitUserSelect: "none", // Safari
        MozUserSelect: "none", // Firefox
        msUserSelect: "none", // IE/Edge
      }}
      className="space-x-4 pointer-events-none select-none" // Disable pointer events and text selection
      onMouseEnter={() => setIsAnimating(false)}
      onMouseLeave={() => setIsAnimating(true)}
      onDragStart={(e) => e.preventDefault()} // Prevent drag start
    >
      <motion.div
        style={{ 
          display: "flex",
          userSelect: "none",
          WebkitUserSelect: "none",
          MozUserSelect: "none",
          msUserSelect: "none",
        }}
        animate={controls}
        className="space-x-5 select-none"
        initial={{ x: 0 }}
        onAnimationComplete={() => {
          controls.set({ x: 0 });
        }}
        drag={false} // Explicitly disable drag
        onDragStart={(e) => e.preventDefault()}
      >
        {duplicatedSlides.map((slide, i) => (
          <div key={i} className="relative group pointer-events-auto select-none" style={{ userSelect: "none" }}> {/* Re-enable for individual items */}
            <div className="w-[320px] h-[300px] border-[#2a2a2a] border-[0.15rem] flex justify-center items-center rounded-xl bg-gradient-to-b from-[#1a1a1a] to-[#0a0a0a] shadow-lg group-hover:shadow-xl transition-all duration-500">
              <img
                src={slide.content}
                alt={`Slide${i + 1}`}
                className="rounded-[1rem] px-3 hover:scale-105 transition-all duration-500 object-cover select-none"
                height={240}
                width={240}
                draggable={false} // Prevent image dragging
                onDragStart={(e) => e.preventDefault()}
                style={{ userSelect: "none", WebkitUserSelect: "none", MozUserSelect: "none", msUserSelect: "none" }}
              />
            </div>
            <div className="mt-5 -ml-2">
              <p className="text-sm inter-400 text-[#8a8a8a] group-hover:text-white transition-colors duration-300 select-none" style={{ userSelect: "none" }}>
                /00{(i % slides.length) + 1} {slide.name.toUpperCase()}
              </p>
            </div>
          </div>
        ))}
      </motion.div>
    </div>
  );
}
