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
const slideWidth = 270;

export function InfiniteCarousel() {
  const controls = useAnimation();
  const isAnimating = useRef(true);

  useEffect(() => {
    const animate = async () => {
      while (!isAnimating.current) {
        await controls.start({
          x: -slides.length * slideWidth * 2,
          transition: { duration: 85, repeat: Infinity, ease: "linear" },
        });
        controls.set({ x: 0 });
      }
    };
    animate();
    return () => {
      isAnimating.current = false;
    };
  }, [controls]);

  const duplicatedSlides = [...slides, ...slides];

  return (
    <div
      style={{
        overflow: "hidden",
        width: slideWidth * slides.length,
        margin: "2rem auto",
        display: "flex",
        flexDirection: "row",
        flexShrink: 0,
      }}
      className="space-x-2"
    >
      <motion.div
        style={{ display: "flex" }}
        animate={controls}
        className="space-x-3"
      >
        {duplicatedSlides.map((slide, i) => (
          <div key={i}>
            <div className="w-[270px] h-[250px] border-[#1e1e1e] border-[0.1rem] flex justify-center items-center rounded-xl">
              <img
                src={slide.content}
                alt={`Slide${i + 1}`}
                className="rounded-[1rem] px-2 hover:scale-105 transition-all duration-500"
                height={200}
                width={200}
              />
            </div>
            <div className="mt-5 -ml-4">
              <p className="text-xs inter-400 gray">
                /00{i < 8 ? i + 1 : i - 8 + 1} {slide.name.toUpperCase()}
              </p>
            </div>
          </div>
        ))}
      </motion.div>
      <motion.div
        style={{ display: "flex" }}
        animate={controls}
        className="space-x-3"
      >
        {duplicatedSlides.map((slide, i) => (
          <div key={i}>
            <div className="w-[270px] h-[250px] border-[#1e1e1e] border-[0.1rem] flex justify-center items-center rounded-xl">
              <img
                src={slide.content}
                alt={`Slide${i + 1}`}
                className="rounded-[1rem] px-2 hover:scale-105 transition-all duration-500"
                height={200}
                width={200}
              />
            </div>
            <div className="mt-5 -ml-4">
              <p className="text-xs inter-400 gray">
                /00{i < 8 ? i + 1 : i - 8 + 1} {slide.name.toUpperCase()}
              </p>
            </div>
          </div>
        ))}
      </motion.div>
    </div>
  );
}
