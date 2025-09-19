import { useState, useEffect } from 'react';
import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtHeader,
  ChainOfThoughtStep,
} from '@/components/ai-elements/chain-of-thought';
import { 
  Upload, 
  Eye, 
  Brain, 
  Zap, 
  Search, 
  CheckCircle, 
  AlertTriangle,
  ImageIcon,
  VideoIcon
} from 'lucide-react';

const AnalysisChainOfThought = ({ isAnalyzing, fileType, result }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState([]);

  const steps = [
    {
      icon: Upload,
      label: "File Upload",
      description: `Processing uploaded ${fileType}...`,
      duration: 800
    },
    {
      icon: fileType === 'image' ? ImageIcon : VideoIcon,
      label: "Media Preprocessing",
      description: "Extracting frames and normalizing pixel values",
      duration: 1200
    },
    {
      icon: Eye,
      label: "Feature Extraction",
      description: "Analyzing visual patterns and anomalies",
      duration: 1500
    },
    {
      icon: Brain,
      label: "AI Model Processing",
      description: "Running through Vision Transformer and GAN classifiers",
      duration: 2000
    },
    {
      icon: Search,
      label: "Pattern Recognition",
      description: "Detecting synthetic artifacts and manipulation traces",
      duration: 1000
    },
    {
      icon: Zap,
      label: "Confidence Calculation",
      description: "Computing final probability scores",
      duration: 600
    }
  ];

  useEffect(() => {
    if (!isAnalyzing) {
      setCurrentStep(0);
      setCompletedSteps([]);
      return;
    }

    let timeoutId;
    let currentIndex = 0;

    const processStep = () => {
      if (currentIndex < steps.length) {
        setCurrentStep(currentIndex);
        
        timeoutId = setTimeout(() => {
          setCompletedSteps(prev => [...prev, currentIndex]);
          currentIndex++;
          if (currentIndex < steps.length) {
            processStep();
          }
        }, steps[currentIndex].duration);
      }
    };

    processStep();

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [isAnalyzing]);

  const getStepStatus = (index) => {
    if (completedSteps.includes(index)) return 'complete';
    if (currentStep === index && isAnalyzing) return 'active';
    return 'pending';
  };

  if (!isAnalyzing && !result) return null;

  return (
    <div className="w-full max-w-2xl mx-auto">
      <ChainOfThought defaultOpen={true}>
        <ChainOfThoughtHeader>
          {isAnalyzing ? "Analyzing for Deepfakes..." : "Analysis Complete"}
        </ChainOfThoughtHeader>
        <ChainOfThoughtContent>
          {steps.map((step, index) => (
            <ChainOfThoughtStep
              key={index}
              icon={step.icon}
              label={step.label}
              description={step.description}
              status={getStepStatus(index)}
            />
          ))}
          
            {/* Analysis Complete with Results */}
            {result && !isAnalyzing && (
              <ChainOfThoughtStep
                icon={result.label === 'fake' ? AlertTriangle : CheckCircle}
                label="Detection Result"
                description={`Analysis shows ${result.confidence}% confidence that this content is ${result.label.toUpperCase()}`}
                status="complete"
              >
                <div className="mt-3 p-4 rounded-lg bg-muted border">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Final Classification:</span>
                      <span className={`text-sm font-bold px-2 py-1 rounded ${
                        result.label === 'fake' 
                          ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' 
                          : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                      }`}>
                        {result.label.toUpperCase()}
                      </span>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Confidence Score:</span>
                        <span className={`text-sm font-bold ${
                          result.label === 'fake' ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'
                        }`}>
                          {result.confidence}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all duration-1000 ${
                            result.label === 'fake' ? 'bg-red-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${result.confidence}%` }}
                        />
                      </div>
                    </div>

                    <div className="pt-2 border-t">
                      <p className="text-xs text-muted-foreground">
                        {result.label === 'fake' 
                          ? "⚠️ Potential deepfake detected. This content may be artificially generated." 
                          : "✅ Content appears authentic with no signs of manipulation detected."
                        }
                      </p>
                    </div>
                  </div>
                </div>
              </ChainOfThoughtStep>
            )}
        </ChainOfThoughtContent>
      </ChainOfThought>
    </div>
  );
};

export default AnalysisChainOfThought;