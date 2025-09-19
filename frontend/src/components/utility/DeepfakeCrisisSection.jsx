import React from "react";
import { motion } from "motion/react";
import { Badge } from "@/components/ui/badge";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Compare } from "@/components/ui/compare";
import { AnimatedCounter } from "@/components/ui/animated-counter";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";
import { UserX, Newspaper, Shield, Scale } from "lucide-react";
import fakeImage from "../../assets/df/fake_image_1.jpg";
import realImage from "../../assets/df/real_image_1.jpg";
import { HoverBorderGradient } from "../ui/hover-border-gradient";
import GradientText from "../ui/GradientText";

const DeepfakeCrisisSection = () => {
  return (
    <motion.section
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      viewport={{ once: true, amount: 0.1 }}
      className="min-h-screen relative overflow-hidden py-20 px-4 bg-black" // Added simple black background
    >
      {/* Removed the complex background elements */}
      
      <div className="max-w-7xl mx-auto relative z-10">
        {/* Title Section */}
        <motion.div
          initial={{ y: -50, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <motion.h2
            className="text-5xl md:text-7xl lg:text-8xl font-bold text-white mb-6"
            style={{ fontFamily: "mubold" }}
          >
            The{" "}
            <GradientText 
              colors={['#ef4444', '#f97316', '#eab308', '#f59e0b']}
              animationSpeed={6}
              className="inline-block"
            >
              <span>Deepfake</span>
            </GradientText>
            {" "}  <span className="block">Crisis</span>

          </motion.h2>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
            viewport={{ once: true }}
            className="text-xl md:text-2xl text-gray-400 max-w-3xl mx-auto leading-relaxed inter-400"
          >
            In an era where reality can be digitally manipulated, our mission is to preserve truth 
            and restore trust in digital media through cutting-edge AI detection technology.
          </motion.p>
        </motion.div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 lg:gap-20 items-center">
          
          {/* Left Column - Statistics and Information */}
          <motion.div
            initial={{ x: -80, opacity: 0 }}
            whileInView={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            viewport={{ once: true }}
            className="space-y-8"
          >
            {/* Statistics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <motion.div
                whileHover={{ scale: 1.05, y: -5 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                className="bg-gradient-to-br from-red-500/15 to-red-600/10 p-6 rounded-2xl border border-red-500/30 backdrop-blur-sm hover:border-red-400/50 transition-colors duration-300"
              >
                <div className="text-3xl md:text-4xl font-bold text-red-400 mb-2 inter-600">
                  <AnimatedCounter to={500000} suffix="+" />
                </div>
                <p className="text-gray-300 text-sm inter-400">Deepfakes detected in 2024</p>
                <div className="mt-2 w-full bg-red-900/30 rounded-full h-1">
                  <motion.div 
                    className="bg-red-500 h-1 rounded-full"
                    initial={{ width: 0 }}
                    whileInView={{ width: "78%" }}
                    transition={{ duration: 1.5, delay: 0.5 }}
                  />
                </div>
              </motion.div>
              
              <motion.div
                whileHover={{ scale: 1.05, y: -5 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                className="bg-gradient-to-br from-orange-500/15 to-orange-600/10 p-6 rounded-2xl border border-orange-500/30 backdrop-blur-sm hover:border-orange-400/50 transition-colors duration-300"
              >
                <div className="text-3xl md:text-4xl font-bold text-orange-400 mb-2 inter-600">
                  <AnimatedCounter to={70} suffix="%" />
                </div>
                <p className="text-gray-300 text-sm inter-400">Report reduced trust in media</p>
                <div className="mt-2 w-full bg-orange-900/30 rounded-full h-1">
                  <motion.div 
                    className="bg-orange-500 h-1 rounded-full"
                    initial={{ width: 0 }}
                    whileInView={{ width: "70%" }}
                    transition={{ duration: 1.5, delay: 0.7 }}
                  />
                </div>
              </motion.div>
            </div>

            {/* Crisis Description */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
              viewport={{ once: true }}
              className="space-y-6"
            >
              <div className="space-y-4">
                <p className="text-lg md:text-xl text-gray-300 leading-relaxed inter-400">
                  The proliferation of sophisticated deepfake technology poses unprecedented threats 
                  to information integrity, democratic processes, and personal privacy. As these 
                  synthetic media become increasingly convincing, the line between authentic and 
                  manipulated content continues to blur.
                </p>

                <TextGenerateEffect
                  words="Our advanced AI detection system employs multi-modal analysis, examining facial inconsistencies, temporal artifacts, and behavioral patterns to identify even the most sophisticated deepfakes with unparalleled precision."
                  className="text-lg text-gray-500 leading-relaxed inter-400 font-normal"
                  filter={true}
                  duration={0.5}
                />
              </div>

              {/* Threat Categories */}
              <div className="flex flex-wrap gap-3">
                <motion.div
                  whileHover={{ scale: 1.1 }}
                  transition={{ type: "spring", stiffness: 400 }}
                >
                  <Badge variant="outline" className="text-red-400 border-red-400 px-4 py-2 text-sm inter-400 flex items-center gap-2">
                    <UserX size={16} />
                    Identity Theft
                  </Badge>
                </motion.div>
                <motion.div
                  whileHover={{ scale: 1.1 }}
                  transition={{ type: "spring", stiffness: 400 }}
                >
                  <Badge variant="outline" className="text-orange-400 border-orange-400 px-4 py-2 text-sm inter-400 flex items-center gap-2">
                    <Newspaper size={16} />
                    Misinformation
                  </Badge>
                </motion.div>
                <motion.div
                  whileHover={{ scale: 1.1 }}
                  transition={{ type: "spring", stiffness: 400 }}
                >
                  <Badge variant="outline" className="text-yellow-400 border-yellow-400 px-4 py-2 text-sm inter-400 flex items-center gap-2">
                    <Shield size={16} />
                    Trust Erosion
                  </Badge>
                </motion.div>
                <motion.div
                  whileHover={{ scale: 1.1 }}
                  transition={{ type: "spring", stiffness: 400 }}
                >
                  <Badge variant="outline" className="text-blue-400 border-blue-400 px-4 py-2 text-sm inter-400 flex items-center gap-2">
                    <Scale size={16} />
                    Legal Implications
                  </Badge>
                </motion.div>
              </div>

              {/* CTA Button */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.7 }}
                viewport={{ once: true }}
                className="pt-4"
              >
                <HoverCard>
                  <HoverCardTrigger asChild>
                    <div className="inline-block">
                      <HoverBorderGradient
                        containerClassName="rounded-full"
                        as="button"
                        className="bg-black hover:bg-gray-900 text-white border-red-500/50 px-8 py-3 text-lg font-semibold transition-all duration-300 inter-600"
                      >
                        <span className="flex items-center space-x-2">
                          <span>Explore Detection Tech</span>
                          <motion.span
                            animate={{ x: [0, 5, 0] }}
                            transition={{ duration: 1.5, repeat: Infinity }}
                          >
                            →
                          </motion.span>
                        </span>
                      </HoverBorderGradient>
                    </div>
                  </HoverCardTrigger>
                  <HoverCardContent className="w-80 bg-gray-900/95 border-red-500/20 backdrop-blur-sm">
                    <div className="space-y-3">
                      <h4 className="font-bold text-white text-lg inter-600">Advanced Detection Methods</h4>
                      <p className="text-sm text-gray-300 leading-relaxed inter-400">
                        Discover how our multi-layered AI approach combines computer vision, 
                        temporal analysis, and behavioral modeling to identify synthetic media 
                        with industry-leading accuracy.
                      </p>
                      <div className="flex space-x-2 pt-2">
                        <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse" style={{ animationDelay: "0.2s" }}></div>
                        <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" style={{ animationDelay: "0.4s" }}></div>
                      </div>
                    </div>
                  </HoverCardContent>
                </HoverCard>
              </motion.div>
            </motion.div>
          </motion.div>

          {/* Right Column - Interactive Comparison */}
          <motion.div
            initial={{ x: 80, opacity: 0 }}
            whileInView={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            viewport={{ once: true }}
            className="flex justify-center"
          >
            <div className="relative group">
              {/* Enhanced glow effect */}
              <div className="absolute -inset-6 bg-gradient-to-r from-red-500/20 via-orange-500/30 to-yellow-500/20 rounded-3xl blur-2xl opacity-60 group-hover:opacity-100 transition-opacity duration-700" />
              <div className="absolute -inset-4 bg-gradient-to-r from-red-500/15 to-orange-500/15 rounded-3xl blur-xl opacity-75 group-hover:opacity-100 transition-opacity duration-500" />
              
              <motion.div
                whileHover={{ scale: 1.02 }}
                transition={{ type: "spring", stiffness: 300 }}
                className="relative"
              >
                <Compare
                  firstImage={fakeImage}
                  secondImage={realImage}
                  className="h-[300px] w-[300px] md:h-[400px] md:w-[400px] lg:h-[500px] lg:w-[500px] rounded-3xl shadow-2xl border-2 border-red-500/30 hover:border-red-400/50 transition-all duration-500"
                  slideMode="drag"
                  autoplay={false}
                  showHandlebar={true}
                />
                
                {/* Floating detection indicators */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, delay: 1.2 }}
                  viewport={{ once: true }}
                  className="absolute top-4 left-4 bg-red-500/90 backdrop-blur-sm text-white px-3 py-1 rounded-full text-xs font-semibold inter-600 shadow-lg"
                >
                  AI Generated
                </motion.div>
                
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, delay: 1.4 }}
                  viewport={{ once: true }}
                  className="absolute top-4 right-4 bg-green-500/90 backdrop-blur-sm text-white px-3 py-1 rounded-full text-xs font-semibold inter-600 shadow-lg"
                >
                  Authentic
                </motion.div>
              </motion.div>

              {/* Enhanced Legend */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.8 }}
                viewport={{ once: true }}
                className="mt-6 flex justify-center space-x-8"
              >
                <div className="flex items-center space-x-3">
                  <motion.div 
                    className="w-4 h-4 bg-red-500 rounded-full shadow-lg"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                  <span className="text-red-400 font-semibold inter-600">Synthetic</span>
                </div>
                <div className="flex items-center space-x-3">
                  <motion.div 
                    className="w-4 h-4 bg-green-500 rounded-full shadow-lg"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 2, repeat: Infinity, delay: 1 }}
                  />
                  <span className="text-green-400 font-semibold inter-600">Authentic</span>
                </div>
              </motion.div>

              <motion.p
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                transition={{ duration: 0.6, delay: 1 }}
                viewport={{ once: true }}
                className="text-center text-gray-500 text-sm mt-4 max-w-sm mx-auto leading-relaxed inter-400"
              >
                <span className="text-gray-400 font-medium"></span>
                Drag the slider to reveal the difference between AI-generated and authentic content
              </motion.p>
            </div>
          </motion.div>
        </div>
      </div>
    </motion.section>
  );
};

export default DeepfakeCrisisSection;