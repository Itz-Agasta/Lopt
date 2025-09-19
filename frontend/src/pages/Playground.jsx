import { useEffect, useRef, useState } from "react";
import { useGlobalContext } from "../hooks/GlobalContext";
import NavBar from "../components/utility/NavBar";
import Footer from "../components/utility/Footer";
import gif from "../assets/waiting.gif";
import Sticker from "../components/utility/h4b";
import { analyzeFile, try_sample } from "../lib/api";
import samples from "../components/static/samples";
import open from "../assets/redirect.svg";
import background from "../assets/bg.png";
import Header from "../components/utility/Header";
import warning from "../assets/warning.png";
import Skeleton, { SkeletonTheme } from "react-loading-skeleton";
import "react-loading-skeleton/dist/skeleton.css";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Badge } from "../components/ui/badge";
import { Progress } from "../components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../components/ui/collapsible";
import AnalysisChainOfThought from "../components/utility/AnalysisChainOfThought";
import { Upload, FileImage, FileVideo, AlertTriangle, CheckCircle, ChevronDown, Eye, EyeOff } from "lucide-react";

const Playground = () => {
  const { isMenuOpen } = useGlobalContext();
  const [uploadedFile, setUploadedFile] = useState(null);
  const [fileType, setFileType] = useState("image");
  const [result, setResult] = useState(null);
  const hiddenFileInput = useRef(null);
  const [position, setPosition] = useState(null);
  const [type, setType] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [showChainOfThought, setShowChainOfThought] = useState(true);

  const handleClick = (e) => {
    hiddenFileInput.current.click();
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    setUploadedFile(file);
    setFileName(file.name);
    if (file.type.startsWith("image/")) setFileType("image");
    else if (file.type.startsWith("video/")) setFileType("video");
    // Reset previous results when new file is uploaded
    setResult(null);
    setType(null);
    setPosition(null);
  };

  const handleSampleSelect = (value) => {
    const index = parseInt(value);
    handleImage(index);
  };

  const handleReset = () => {
    setUploadedFile(null);
    setFileName(null);
    setResult(null);
    setType(null);
    setPosition(null);
    setSubmitting(false);
    setShowChainOfThought(true);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setResult(null); // Clear previous results
    setShowChainOfThought(true); // Show chain of thought during analysis
    console.log("Submit Started!");

    try {
      if (position !== null && type) {
        console.log("Using sample - Position:", position, "Type:", type);
        const response = await try_sample(position, type);
        // Add a small delay to show the complete analysis
        setTimeout(() => {
          setResult(response);
          setShowChainOfThought(false); // Hide chain of thought after results
        }, 6000); // Wait for chain of thought to complete
        console.log(response);
      } else if (uploadedFile) {
        const response = await analyzeFile(uploadedFile);
        // Add a small delay to show the complete analysis
        setTimeout(() => {
          setResult(response);
          setShowChainOfThought(false); // Hide chain of thought after results
        }, 6000); // Wait for chain of thought to complete
        console.log(response);
      }
    } catch (err) {
      console.log("Error!", err);
      setResult(null);
      setShowChainOfThought(false);
    } finally {
      setTimeout(() => {
        setSubmitting(false);
      }, 6000); // Stop submitting after chain of thought completes
    }
  };

  const handleImage = async (index) => {
    setResult(null);
    setShowChainOfThought(true);
    const sample = samples[index];
    setFileName(sample.name);
    
    // Parse the sample name to extract information
    const name = sample.name.split("_");
    console.log("Sample name parts:", name);
    
    // Extract index from name (fake_image_1 -> index 0, fake_image_2 -> index 1, etc.)
    const sampleIndex = parseInt(name[2]) - 1;
    setPosition(sampleIndex);
    
    // Determine file type and set it
    if (name.includes("image")) {
      setFileType("image");
      setType("image");
    } else if (name.includes("video")) {
      setFileType("video");
      setType("video");
    }
    
    console.log("Position set to:", sampleIndex);
    console.log("Type set to:", name.includes("image") ? "image" : "video");
    console.log("File type set to:", name.includes("image") ? "image" : "video");
  };

  return (
    <>
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900">
        <NavBar />
        <Header />
        {isMenuOpen ? (
          <div className="h-screen w-screen z-20 fixed inset-0 backdrop-brightness-75"></div>
        ) : (
          <></>
        )}
        <Sticker />
        
        <div className="container mx-auto px-4 py-8 pt-24">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-white via-gray-300 to-gray-500 bg-clip-text text-transparent mb-4">
              Detect Deepfakes with AI
            </h1>
            <h2 className="text-xl md:text-2xl text-gray-400 mb-4">
              [LOPT] - Advanced Vision Analysis
            </h2>
            <p className="text-gray-300 text-lg max-w-3xl mx-auto">
              Upload your media files and let our advanced AI models analyze them for deepfake detection.
              Using Vision Transformers and GAN classifiers for accurate results.
            </p>
          </div>

          <div className="max-w-4xl mx-auto space-y-8">
            {/* Upload Card */}
            <Card className="bg-gray-800/50 border-gray-700 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Upload className="w-5 h-5" />
                  Upload Media for Analysis
                </CardTitle>
                <CardDescription className="text-gray-400">
                  Select an image or video file to analyze for deepfake detection
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* File Upload Area */}
                <div 
                  onClick={handleClick}
                  className="border-2 border-dashed border-gray-600 rounded-lg p-8 hover:border-gray-500 transition-colors cursor-pointer bg-gray-800/30"
                >
                  <div className="text-center">
                    {fileName ? (
                      <div className="space-y-2">
                        <div className="flex items-center justify-center text-green-400">
                          {fileType === 'image' ? <FileImage className="w-8 h-8" /> : <FileVideo className="w-8 h-8" />}
                        </div>
                        <p className="text-white font-medium">{fileName}</p>
                        <p className="text-gray-400 text-sm">Click to change file</p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <Upload className="w-12 h-12 text-gray-400 mx-auto" />
                        <p className="text-white font-medium">Drop your file here or click to browse</p>
                        <p className="text-gray-400 text-sm">Supports images and videos</p>
                      </div>
                    )}
                  </div>
                  <input
                    type="file"
                    accept="image/*,video/*"
                    ref={hiddenFileInput}
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </div>

                {/* File Type Toggle */}
                <div className="flex justify-center">
                  <div className="flex bg-gray-700 rounded-lg p-1">
                    <Button
                      variant={fileType === "image" ? "default" : "ghost"}
                      size="sm"
                      className={`${fileType === "image" ? "bg-white text-black" : "text-gray-300 hover:text-white"}`}
                    >
                      <FileImage className="w-4 h-4 mr-2" />
                      Image
                    </Button>
                    <Button
                      variant={fileType === "video" ? "default" : "ghost"}
                      size="sm"
                      className={`${fileType === "video" ? "bg-white text-black" : "text-gray-300 hover:text-white"}`}
                    >
                      <FileVideo className="w-4 h-4 mr-2" />
                      Video
                    </Button>
                  </div>
                </div>

                {/* Sample Files Dropdown */}
                <div className="space-y-4">
                  <h3 className="text-white font-medium">Or try a sample file:</h3>
                  <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
                    <Select onValueChange={handleSampleSelect}>
                      <SelectTrigger className="w-full sm:w-[300px] bg-gray-700/50 border-gray-600 text-white">
                        <SelectValue placeholder="Choose a sample file..." />
                      </SelectTrigger>
                      <SelectContent className="bg-gray-800 border-gray-600">
                        {samples.map((item, idx) => (
                          <SelectItem 
                            key={idx} 
                            value={idx.toString()}
                            className="text-gray-200 hover:bg-gray-700 focus:bg-gray-700"
                          >
                            <div className="flex items-center justify-between w-full">
                              <span>{item.name}</span>
                              <Badge 
                                variant="outline" 
                                className="ml-2 text-xs border-gray-500 text-gray-400"
                              >
                                {item.name.includes('image') ? 'IMG' : 'VID'}
                              </Badge>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    
                    {fileName && samples.some((_, idx) => idx.toString() === position?.toString()) && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          const sample = samples[position];
                          window.open(sample.source, "_blank", "noopener, noreferrer");
                        }}
                        className="text-gray-400 hover:text-white flex items-center gap-2"
                      >
                        <img src={open} height={16} width={16} alt="Open" />
                        View Source
                      </Button>
                    )}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex justify-center gap-4">
                  {!result && !submitting && (uploadedFile || (position !== null && type)) && (
                    <Button 
                      onClick={handleSubmit}
                      className="bg-red-600 hover:bg-red-700 text-white px-8 py-3 text-lg"
                    >
                      Analyze for Deepfakes
                    </Button>
                  )}
                  
                  {(result || uploadedFile || fileName) && !submitting && (
                    <Button 
                      onClick={handleReset}
                      variant="outline"
                      className="border-gray-600 text-gray-300 hover:bg-gray-700 px-6 py-3"
                    >
                      Start Over
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Analysis Process */}
            {(submitting && showChainOfThought) && (
              <AnalysisChainOfThought 
                isAnalyzing={submitting} 
                fileType={fileType} 
                result={null}
              />
            )}

            {/* Results Section */}
            {result && (
              <Card className="bg-gray-800/50 border-gray-700 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    {result.label === "fake" ? (
                      <AlertTriangle className="w-5 h-5 text-red-500" />
                    ) : (
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    )}
                    Detection Results
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Classification:</span>
                    <Badge 
                      variant={result.label === "fake" ? "destructive" : "default"}
                      className={`${result.label === "fake" ? "bg-red-600" : "bg-green-600"} text-white`}
                    >
                      {result.label.toUpperCase()}
                    </Badge>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-300">Confidence:</span>
                      <span className={`font-bold ${result.label === "fake" ? "text-red-400" : "text-green-400"}`}>
                        {result.confidence}%
                      </span>
                    </div>
                    <Progress 
                      value={result.confidence} 
                      className="h-3"
                    />
                  </div>
                  
                  <div className="mt-4 p-4 bg-gray-700/50 rounded-lg">
                    <p className="text-gray-300 text-sm">
                      {result.label === "fake" 
                        ? "⚠️ This content appears to be artificially generated or manipulated. Exercise caution when sharing or trusting this media."
                        : "✅ This content appears to be authentic with no signs of artificial generation or manipulation detected."
                      }
                    </p>
                  </div>

                  {/* Chain of Thought Toggle */}
                  <div className="pt-4 border-t border-gray-600">
                    <Collapsible open={showChainOfThought} onOpenChange={setShowChainOfThought}>
                      <CollapsibleTrigger asChild>
                        <Button 
                          variant="ghost" 
                          className="text-gray-300 hover:text-white p-0 h-auto font-normal"
                        >
                          <div className="flex items-center gap-2">
                            {showChainOfThought ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                            {showChainOfThought ? "Hide" : "Show"} Analysis Process
                            <ChevronDown className={`w-4 h-4 transition-transform ${showChainOfThought ? "rotate-180" : ""}`} />
                          </div>
                        </Button>
                      </CollapsibleTrigger>
                      <CollapsibleContent className="mt-4">
                        <AnalysisChainOfThought 
                          isAnalyzing={false} 
                          fileType={fileType} 
                          result={result}
                        />
                      </CollapsibleContent>
                    </Collapsible>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Warning Notice */}
            <Card className="bg-orange-900/20 border-orange-700/50">
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <img src={warning} height={24} width={24} alt="Warning" />
                  <p className="text-orange-200 text-sm">
                    We are currently facing issues with our sample routes. Please upload files from your local machine for the best experience.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
        
        <Footer />
      </div>
    </>
  );
};

export default Playground;
