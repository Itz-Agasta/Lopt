import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { motion } from "motion/react";
import { NavLink } from "react-router";
import model1 from "../../assets/virtus.png";
import model2 from "../../assets/scarlett.png";
import StatisticsShowcase from "./StatisticsShowcase";
import { useState, useEffect } from "react";
import DeepfakeCrisisSection from "./DeepfakeCrisisSection";

const ScientificSections = () => {
  const [accuracyProgress, setAccuracyProgress] = useState(0);
  const [f1Progress, setF1Progress] = useState(0);
  const [confidenceThreshold, setConfidenceThreshold] = useState([95]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedModel, setSelectedModel] = useState("virtus");

  useEffect(() => {
    const timer1 = setTimeout(() => setAccuracyProgress(99.2), 1000);
    const timer2 = setTimeout(() => setF1Progress(95.8), 1200);
    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, []);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6, ease: "easeOut" },
    },
  };

  return (
    <TooltipProvider>
      <motion.div
        className="w-full bg-black min-h-screen py-16"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.1 }}
      >
        <div className="max-w-7xl mx-auto px-4 space-y-16">
          

          {/* The Deepfake Crisis - Reimagined */}
          <DeepfakeCrisisSection />

          {/* Interactive Statistics Showcase */}
          <StatisticsShowcase />




          {/* Interactive Model Comparison */}
          <motion.section variants={itemVariants} className="px-4">
            <div className="flex items-center justify-between mb-8">
              <h3 className="text-3xl md:text-4xl font-bold text-white">
                Interactive Model Explorer
              </h3>
              
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="outline" className="border-red-500 text-red-500 hover:bg-red-500 hover:text-white">
                    Compare Models
                  </Button>
                </DialogTrigger>
                <DialogContent className="bg-gray-900 border-gray-700 max-w-4xl">
                  <DialogHeader>
                    <DialogTitle className="text-white">Model Performance Comparison</DialogTitle>
                    <DialogDescription className="text-gray-300">
                      Detailed comparison of Virtus and Scarlet model architectures
                    </DialogDescription>
                  </DialogHeader>
                  <div className="grid md:grid-cols-2 gap-6 mt-4">
                    <div className="space-y-4">
                      <h4 className="text-lg font-semibold text-white">Virtus (Image)</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-300">Accuracy</span>
                          <span className="text-green-500">99.2%</span>
                        </div>
                        <Progress value={99.2} className="h-2" />
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-300">F1 Score</span>
                          <span className="text-green-500">0.9920</span>
                        </div>
                        <Progress value={99.2} className="h-2" />
                      </div>
                    </div>
                    <div className="space-y-4">
                      <h4 className="text-lg font-semibold text-white">Scarlet (Video)</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-300">Accuracy</span>
                          <span className="text-blue-500">96.0%</span>
                        </div>
                        <Progress value={96} className="h-2" />
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-300">F1 Score</span>
                          <span className="text-blue-500">0.958</span>
                        </div>
                        <Progress value={95.8} className="h-2" />
                      </div>
                    </div>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
            
            <Tabs value={selectedModel} onValueChange={setSelectedModel} className="w-full max-w-6xl mx-auto">
              <TabsList className="grid w-full grid-cols-2 mb-8 bg-gray-900 h-12">
                <TabsTrigger value="virtus" className="text-sm md:text-lg data-[state=active]:bg-red-500">
                  Virtus - Image Detection
                </TabsTrigger>
                <TabsTrigger value="scarlet" className="text-sm md:text-lg data-[state=active]:bg-blue-500">
                  Scarlet - Video Analysis
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="virtus" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
                  <motion.div
                    initial={{ x: -50, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ duration: 0.5 }}
                  >
                    <Card className="bg-gray-900 border-gray-700 hover:border-red-500 transition-all duration-300">
                      <CardHeader>
                        <CardTitle className="text-xl md:text-2xl text-white flex items-center gap-2">
                          <Badge className="bg-red-500">ViT</Badge>
                          Virtus Model
                          
                          <Popover>
                            <PopoverTrigger asChild>
                              <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                                ℹ️
                              </Button>
                            </PopoverTrigger>
                            <PopoverContent className="w-80 bg-gray-900 border-gray-700">
                              <div className="space-y-2">
                                <h4 className="font-medium text-white">Vision Transformer Architecture</h4>
                                <p className="text-sm text-gray-300">
                                  Based on facebook/deit-base-distilled-patch16-224 with custom classification head
                                  for binary deepfake detection.
                                </p>
                              </div>
                            </PopoverContent>
                          </Popover>
                        </CardTitle>
                        <CardDescription className="text-gray-300">
                          Vision Transformer for image-level deepfake detection
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-300">Live Accuracy</span>
                            <span className="text-white">{accuracyProgress}%</span>
                          </div>
                          <Progress value={accuracyProgress} className="h-3" />
                        </div>
                        
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-300">Confidence Threshold</span>
                            <span className="text-white">{confidenceThreshold[0]}%</span>
                          </div>
                          <Slider
                            value={confidenceThreshold}
                            onValueChange={setConfidenceThreshold}
                            max={100}
                            min={50}
                            step={1}
                            className="w-full"
                          />
                        </div>
                        
                        <div className="pt-4 flex flex-wrap gap-2">
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Badge variant="outline" className="text-xs cursor-help">facebook/deit-base</Badge>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>Base architecture from Facebook AI</p>
                            </TooltipContent>
                          </Tooltip>
                          <Badge variant="outline" className="text-xs">190k samples</Badge>
                          <Badge variant="outline" className="text-xs">Binary Classification</Badge>
                        </div>
                        
                        <Sheet>
                          <SheetTrigger asChild>
                            <Button className="w-full bg-red-500 hover:bg-red-600">
                              🔬 Explore Architecture
                            </Button>
                          </SheetTrigger>
                          <SheetContent className="bg-gray-900 border-gray-700">
                            <SheetHeader>
                              <SheetTitle className="text-white">Virtus Architecture Details</SheetTitle>
                              <SheetDescription className="text-gray-300">
                                Deep dive into the Vision Transformer architecture
                              </SheetDescription>
                            </SheetHeader>
                            <div className="mt-6 space-y-4">
                              <div className="grid grid-cols-2 gap-4">
                                <div>
                                  <p className="text-sm font-medium text-white">Input Size</p>
                                  <p className="text-sm text-gray-300">224x224 RGB</p>
                                </div>
                                <div>
                                  <p className="text-sm font-medium text-white">Patch Size</p>
                                  <p className="text-sm text-gray-300">16x16</p>
                                </div>
                                <div>
                                  <p className="text-sm font-medium text-white">Hidden Dim</p>
                                  <p className="text-sm text-gray-300">768</p>
                                </div>
                                <div>
                                  <p className="text-sm font-medium text-white">Layers</p>
                                  <p className="text-sm text-gray-300">12</p>
                                </div>
                              </div>
                              <div className="space-y-2">
                                <p className="text-sm font-medium text-white">Training Process</p>
                                <p className="text-sm text-gray-300">
                                  Fine-tuned on curated deepfake dataset with data augmentation and 
                                  stratified sampling for balanced representation.
                                </p>
                              </div>
                            </div>
                          </SheetContent>
                        </Sheet>
                      </CardContent>
                    </Card>
                  </motion.div>
                  
                  <motion.div 
                    className="flex justify-center"
                    whileHover={{ scale: 1.05, rotate: 2 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <img src={model1} alt="Virtus Model" className="max-w-full w-80 md:w-96 h-auto rounded-lg shadow-lg border-2 border-red-500/30" />
                  </motion.div>
                </div>
              </TabsContent>
              
              <TabsContent value="scarlet" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
                  <motion.div
                    initial={{ x: -50, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ duration: 0.5 }}
                  >
                    <Card className="bg-gray-900 border-gray-700 hover:border-blue-500 transition-all duration-300">
                      <CardHeader>
                        <CardTitle className="text-xl md:text-2xl text-white flex items-center gap-2">
                          <Badge className="bg-blue-500">TimeSformer</Badge>
                          Scarlet Model
                          
                          <Popover>
                            <PopoverTrigger asChild>
                              <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                                
                              </Button>
                            </PopoverTrigger>
                            <PopoverContent className="w-80 bg-gray-900 border-gray-700">
                              <div className="space-y-2">
                                <h4 className="font-medium text-white">TimeSformer Architecture</h4>
                                <p className="text-sm text-gray-300">
                                  Temporal transformer that processes video frames with spatial-temporal attention
                                  for robust video deepfake detection.
                                </p>
                              </div>
                            </PopoverContent>
                          </Popover>
                        </CardTitle>
                        <CardDescription className="text-gray-300">
                          Temporal transformer for video-level deepfake analysis
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-300">Video Accuracy</span>
                            <span className="text-white">96.0%</span>
                          </div>
                          <Progress value={96} className="h-3" />
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-300">F1 Score</span>
                            <span className="text-white">0.958</span>
                          </div>
                          <Progress value={f1Progress} className="h-3" />
                        </div>
                        <div className="pt-4 flex flex-wrap gap-2">
                          <Badge variant="outline" className="text-xs">facebook/timesformer</Badge>
                          <Badge variant="outline" className="text-xs">FaceForensics++</Badge>
                          <Badge variant="outline" className="text-xs">Temporal Attention</Badge>
                        </div>
                        
                        <Sheet>
                          <SheetTrigger asChild>
                            <Button className="w-full bg-blue-500 hover:bg-blue-600">
                              Explore Video Analysis
                            </Button>
                          </SheetTrigger>
                          <SheetContent className="bg-gray-900 border-gray-700">
                            <SheetHeader>
                              <SheetTitle className="text-white">Scarlet Video Analysis</SheetTitle>
                              <SheetDescription className="text-gray-300">
                                Understanding temporal deepfake detection
                              </SheetDescription>
                            </SheetHeader>
                            <div className="mt-6 space-y-4">
                              <div className="grid grid-cols-2 gap-4">
                                <div>
                                  <p className="text-sm font-medium text-white">Frame Rate</p>
                                  <p className="text-sm text-gray-300">8 FPS sampling</p>
                                </div>
                                <div>
                                  <p className="text-sm font-medium text-white">Sequence Length</p>
                                  <p className="text-sm text-gray-300">8 frames</p>
                                </div>
                                <div>
                                  <p className="text-sm font-medium text-white">Resolution</p>
                                  <p className="text-sm text-gray-300">224x224</p>
                                </div>
                                <div>
                                  <p className="text-sm font-medium text-white">Attention</p>
                                  <p className="text-sm text-gray-300">Spatial-Temporal</p>
                                </div>
                              </div>
                            </div>
                          </SheetContent>
                        </Sheet>
                      </CardContent>
                    </Card>
                  </motion.div>
                  
                  <motion.div 
                    className="flex justify-center"
                    whileHover={{ scale: 1.05, rotate: -2 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <img src={model2} alt="Scarlet Model" className="max-w-full w-80 md:w-96 h-auto rounded-lg shadow-lg border-2 border-blue-500/30" />
                  </motion.div>
                </div>
              </TabsContent>
            </Tabs>
          </motion.section>

          {/* Interactive Technical Specifications */}
          <motion.section variants={itemVariants} className="px-4">
            <div className="flex items-center justify-between mb-12">
              <h3 className="text-3xl md:text-4xl font-bold text-white">
                Technical Specifications
              </h3>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-300">Advanced View</span>
                <Switch checked={showAdvanced} onCheckedChange={setShowAdvanced} />
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
              <motion.div whileHover={{ y: -5 }}>
                <Card className="bg-gray-900 border-gray-700 hover:border-red-500 transition-colors duration-300 h-full">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2">
                      Architecture
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                        
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Technology stack and frameworks</p>
                        </TooltipContent>
                      </Tooltip>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Framework</span>
                      <Badge>PyTorch</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Backend</span>
                      <Badge>FastAPI</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Frontend</span>
                      <Badge>React + Vite</Badge>
                    </div>
                    {showAdvanced && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-gray-300">Deployment</span>
                          <Badge>Docker</Badge>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-300">CI/CD</span>
                          <Badge>GitHub Actions</Badge>
                        </div>
                      </>
                    )}
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div whileHover={{ y: -5 }}>
                <Card className="bg-gray-900 border-gray-700 hover:border-blue-500 transition-colors duration-300 h-full">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2">
                      Performance
                      <HoverCard>
                        <HoverCardTrigger asChild>
                          <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                          
                          </Button>
                        </HoverCardTrigger>
                        <HoverCardContent className="w-80 bg-gray-900 border-gray-700">
                          <div className="space-y-2">
                            <h4 className="text-sm font-semibold text-white">Performance Metrics</h4>
                            <p className="text-sm text-gray-300">
                              Real-time inference capabilities with optimized model loading and GPU acceleration.
                            </p>
                          </div>
                        </HoverCardContent>
                      </HoverCard>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Inference Speed</span>
                      <Badge variant="outline">~200ms</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Model Size</span>
                      <Badge variant="outline">~300MB</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Batch Support</span>
                      <Badge variant="outline">Yes</Badge>
                    </div>
                    {showAdvanced && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-gray-300">GPU Memory</span>
                          <Badge variant="outline">2GB</Badge>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-300">Throughput</span>
                          <Badge variant="outline">100 img/min</Badge>
                        </div>
                      </>
                    )}
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div whileHover={{ y: -5 }}>
                <Card className="bg-gray-900 border-gray-700 hover:border-green-500 transition-colors duration-300 h-full">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2">
                      Research Impact
                      <Dialog>
                        <DialogTrigger asChild>
                          <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                          
                          </Button>
                        </DialogTrigger>
                        <DialogContent className="bg-gray-900 border-gray-700">
                          <DialogHeader>
                            <DialogTitle className="text-white">Research Publications</DialogTitle>
                            <DialogDescription className="text-gray-300">
                              Academic contributions and citations
                            </DialogDescription>
                          </DialogHeader>
                          <div className="space-y-4 mt-4">
                            <div className="p-4 bg-gray-800 rounded-lg">
                              <h4 className="font-medium text-white">FaceForensics++</h4>
                              <p className="text-sm text-gray-300">Rossler et al., ICCV 2019</p>
                            </div>
                            <div className="p-4 bg-gray-800 rounded-lg">
                              <h4 className="font-medium text-white">TimeSformer</h4>
                              <p className="text-sm text-gray-300">Bertasius et al., 2021</p>
                            </div>
                          </div>
                        </DialogContent>
                      </Dialog>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Publications</span>
                      <Badge variant="outline">4+ Papers</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Datasets</span>
                      <Badge variant="outline">2 Curated</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Open Source</span>
                      <Badge variant="outline">MIT License</Badge>
                    </div>
                    {showAdvanced && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-gray-300">HuggingFace</span>
                          <Badge variant="outline">2 Models</Badge>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-300">Citations</span>
                          <Badge variant="outline">50+</Badge>
                        </div>
                      </>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            </div>
          </motion.section>

          {/* Interactive FAQ Section */}
          <motion.section variants={itemVariants} className="px-4">
            <h3 className="text-3xl md:text-4xl font-bold text-white text-center mb-12">
              Frequently Asked Questions
            </h3>
            
            <Accordion type="single" collapsible className="w-full max-w-4xl mx-auto">
              <AccordionItem value="item-1" className="border-gray-700">
                <AccordionTrigger className="text-white hover:text-red-500 text-left">
                  How accurate are the deepfake detection models?
                </AccordionTrigger>
                <AccordionContent className="text-gray-300">
                  Our Virtus model achieves 99.2% accuracy on image detection, while Scarlet achieves 96% accuracy for video analysis. 
                  Both models are trained on large, diverse datasets and employ state-of-the-art transformer architectures for robust performance.
                </AccordionContent>
              </AccordionItem>
              
              <AccordionItem value="item-2" className="border-gray-700">
                <AccordionTrigger className="text-white hover:text-red-500 text-left">
                  What makes LOPT different from other detection systems?
                </AccordionTrigger>
                <AccordionContent className="text-gray-300">
                  LOPT combines dual-modal detection (images and videos) with modern transformer architectures. 
                  Unlike traditional CNN-based approaches, our Vision Transformer and TimeSformer models capture 
                  long-range dependencies and subtle artifacts that are often missed by conventional methods.
                </AccordionContent>
              </AccordionItem>
              
              <AccordionItem value="item-3" className="border-gray-700">
                <AccordionTrigger className="text-white hover:text-red-500 text-left">
                  Can I use LOPT for commercial applications?
                </AccordionTrigger>
                <AccordionContent className="text-gray-300">
                  Yes! LOPT is released under the MIT license, making it suitable for both research and commercial use. 
                  The models are available on HuggingFace Hub, and the complete codebase is open source on GitHub.
                </AccordionContent>
              </AccordionItem>
              
              <AccordionItem value="item-4" className="border-gray-700">
                <AccordionTrigger className="text-white hover:text-red-500 text-left">
                  How do I integrate LOPT into my existing workflow?
                </AccordionTrigger>
                <AccordionContent className="text-gray-300">
                  LOPT provides multiple integration options: a REST API for web services, Python packages for direct integration, 
                  Docker containers for easy deployment, and pre-trained models available through HuggingFace Transformers library.
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </motion.section>

          {/* Interactive Call to Action */}
          <motion.section variants={itemVariants} className="text-center px-4">
            <Card className="bg-gradient-to-r from-red-900/20 to-orange-900/20 border-red-500/30 max-w-4xl mx-auto">
              <CardContent className="p-6 md:p-8">
                <motion.h3 
                  className="text-2xl md:text-3xl font-bold text-white mb-4"
                  whileHover={{ scale: 1.05 }}
                >
                  Ready to Detect Deepfakes?
                </motion.h3>
                <p className="text-gray-300 mb-6 text-base md:text-lg">
                  Try our advanced detection models or explore the research behind them.
                </p>
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <NavLink to="/playground">
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                      <Button size="lg" className="bg-red-500 hover:bg-red-600 text-white w-full sm:w-auto">
                        Try Demo
                      </Button>
                    </motion.div>
                  </NavLink>
                  <NavLink to="/models">
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                      <Button variant="outline" size="lg" className="border-red-500 text-red-500 hover:bg-red-500 hover:text-white w-full sm:w-auto">
                        Explore Models
                      </Button>
                    </motion.div>
                  </NavLink>
                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                    <Button variant="outline" size="lg" className="border-gray-500 text-gray-300 hover:bg-gray-700 w-full sm:w-auto">
                      <a href="https://github.com/Itz-Agasta/lopt" target="_blank" rel="noopener noreferrer">
                        View Source
                      </a>
                    </Button>
                  </motion.div>
                </div>
              </CardContent>
            </Card>
          </motion.section>
        </div>
      </motion.div>
    </TooltipProvider>
  );
};

export default ScientificSections;