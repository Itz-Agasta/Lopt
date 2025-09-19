import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";

const StatisticsShowcase = () => {
  const [counts, setCounts] = useState({
    deepfakes: 0,
    accuracy: 0,
    papers: 0,
    models: 0,
  });

  useEffect(() => {
    const timer = setTimeout(() => {
      setCounts({
        deepfakes: 500000,
        accuracy: 99.2,
        papers: 4,
        models: 2,
      });
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  const stats = [
    {
      value: counts.deepfakes.toLocaleString(),
      label: "Deepfakes Detected in 2024",
      suffix: "+",
      color: "text-red-500",
      bgColor: "bg-red-500/10",
    },
    {
      value: counts.accuracy,
      label: "Maximum Accuracy Achieved",
      suffix: "%",
      color: "text-green-500",
      bgColor: "bg-green-500/10",
    },
    {
      value: counts.papers,
      label: "Research Papers Referenced",
      suffix: "+",
      color: "text-blue-500",
      bgColor: "bg-blue-500/10",
    },
    {
      value: counts.models,
      label: "Production-Ready Models",
      suffix: "",
      color: "text-purple-500",
      bgColor: "bg-purple-500/10",
    },
  ];

  return (
    <div className="w-full py-16 bg-gradient-to-b from-black to-gray-900">
      <div className="max-w-7xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <Badge variant="outline" className="mb-4 text-yellow-500 border-yellow-500 bg-yellow-500/10">
            Research Impact
          </Badge>
          <h2 className="text-3xl md:text-5xl font-bold text-white mb-4">
            By the Numbers
          </h2>
          <p className="text-gray-300 text-lg max-w-2xl mx-auto">
            Our research and models are making a real impact in the fight against digital deception
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
            >
              <Card className={`bg-gray-900 border-gray-700 hover:border-gray-600 transition-all duration-300 ${stat.bgColor} hover:scale-105`}>
                <CardContent className="p-6 text-center">
                  <div className={`text-4xl md:text-5xl font-bold ${stat.color} mb-2`}>
                    {typeof stat.value === 'number' && stat.value > 1000
                      ? Math.floor(stat.value / 1000) + 'k'
                      : stat.value}
                    <span className="text-2xl">{stat.suffix}</span>
                  </div>
                  <p className="text-gray-300 text-sm font-medium">
                    {stat.label}
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default StatisticsShowcase;