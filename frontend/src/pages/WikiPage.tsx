import React from 'react';
import { Book, Keyboard, Target, Zap, Activity, Layers, Wind, AlignLeft, BarChart2 } from 'lucide-react';

const WikiPage: React.FC = () => {
  return (
    <div className="w-full max-w-[1600px] mx-auto p-6 flex flex-col gap-6"> {/* Back to large width */}
      {/* Header */}
      <div className="flex items-center gap-4 pb-2 border-b border-gray-800">
        <div className="p-3 bg-indigo-500/10 rounded-xl">
          <Book className="text-indigo-400" size={28} />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Wiki</h1>
          <p className="text-gray-400 text-sm">Understanding the mechanics behind the metrics.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6"> {/* Back to 2 cols */}
        
        {/* Core Concepts */}
        <section className="bg-gray-900 rounded-xl border border-gray-800 p-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <Zap className="text-blue-400" size={24} />
            </div>
            <h2 className="text-2xl font-bold text-white">Core Metrics</h2>
          </div>
          
          <div className="space-y-8">
            <div>
              <div className="flex items-baseline gap-3 mb-1">
                <h3 className="text-xl font-semibold text-white">WPM</h3>
                <span className="text-sm text-gray-600 font-mono italic">/wərdz pər ˈmɪnɪt/</span>
              </div>
              <p className="text-gray-400 text-lg leading-relaxed">
                Standard measure of typing speed. Calculated as (Correct Characters / 5) / Time in Minutes. 
                Penalizes uncorrected errors heavily.
              </p>
            </div>

            <div>
              <div className="flex items-baseline gap-3 mb-1">
                <h3 className="text-xl font-semibold text-white">Raw WPM</h3>
                <span className="text-sm text-gray-600 font-mono italic">/rɔː wərdz pər ˈmɪnɪt/</span>
              </div>
              <p className="text-gray-400 text-lg leading-relaxed">
                Gross speed calculation irrespective of errors. (Total Keystrokes / 5) / Time. 
                A large gap between Raw WPM and WPM indicates inefficient typing due to backspacing.
              </p>
            </div>

            <div>
              <div className="flex items-baseline gap-3 mb-1">
                <h3 className="text-xl font-semibold text-white">Accuracy</h3>
                <span className="text-sm text-gray-600 font-mono italic">/ˈækjərəsi/</span>
              </div>
              <p className="text-gray-400 text-lg leading-relaxed">
                Percentage of correct keystrokes out of total keystrokes. Accuracy is foundational; speed built on 
                poor accuracy is fragile.
              </p>
            </div>
          </div>
        </section>

        {/* Advanced Mechanics */}
        <section className="bg-gray-900 rounded-xl border border-gray-800 p-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-emerald-500/10 rounded-lg">
              <Activity className="text-emerald-400" size={24} />
            </div>
            <h2 className="text-2xl font-bold text-white">Flow Mechanics</h2>
          </div>

          <div className="space-y-8">
            <div>
              <div className="flex items-center gap-3 mb-1">
                <Layers size={20} className="text-indigo-400" />
                <h3 className="text-xl font-semibold text-white">Rollover</h3>
                <span className="text-sm text-gray-600 font-mono italic">/ˈrəʊlˌəʊvər/</span>
              </div>
              <p className="text-gray-400 text-lg leading-relaxed">
                The percentage of keystrokes where the next key is pressed <em>before</em> the previous key is released. 
                High rollover indicates fluid, legato typing rather than staccato "pecking." It is essential for high-speed typing.
              </p>
            </div>

            <div>
              <div className="flex items-center gap-3 mb-1">
                <Wind size={20} className="text-cyan-400" />
                <h3 className="text-xl font-semibold text-white">Flow State</h3>
                <span className="text-sm text-gray-600 font-mono italic">/fləʊ steɪt/</span>
              </div>
              <p className="text-gray-400 text-lg leading-relaxed">
                The percentage of typing intervals that are smooth and rhythmic, as opposed to "spikes" caused by hesitations 
                or difficult transitions. A higher flow score means you are maintaining momentum.
              </p>
            </div>

            <div>
              <div className="flex items-center gap-3 mb-1">
                <AlignLeft size={20} className="text-rose-400" />
                <h3 className="text-xl font-semibold text-white">Avg Chunk</h3>
                <span className="text-sm text-gray-600 font-mono italic">/ævərɪdʒ tʃʌŋk/</span>
              </div>
              <p className="text-gray-400 text-lg leading-relaxed">
                The average number of characters typed in a single burst without a significant pause. Experts "chunk" entire words 
                or bigrams (e.g., "ing", "tion") as single motor units.
              </p>
            </div>
          </div>
        </section>

        {/* Technical Terms */}
        <section className="bg-gray-900 rounded-xl border border-gray-800 p-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-purple-500/10 rounded-lg">
              <BarChart2 className="text-purple-400" size={24} />
            </div>
            <h2 className="text-2xl font-bold text-white">Technical Analysis</h2>
          </div>

          <div className="space-y-8">
            <div>
              <div className="flex items-baseline gap-3 mb-1">
                <h3 className="text-xl font-semibold text-white">IKI</h3>
                <span className="text-sm text-gray-600 font-mono italic">/ˌaɪ keɪ ˈaɪ/</span>
              </div>
              <p className="text-gray-400 text-lg leading-relaxed">
                <strong>Inter-Key Interval.</strong> The time in milliseconds between two consecutive keystrokes. 
                Lower IKI = higher speed. Variance in IKI is used to measure consistency.
              </p>
            </div>

            <div>
              <div className="flex items-baseline gap-3 mb-1">
                <h3 className="text-xl font-semibold text-white">KSPC</h3>
                <span className="text-sm text-gray-600 font-mono italic">/keɪ ɛs piː siː/</span>
              </div>
              <p className="text-gray-400 text-lg leading-relaxed">
                <strong>Keystrokes Per Character.</strong> Ideally 1.0. A value of 1.1 means you type 1.1 keys to produce 
                1 correct character (due to errors and backspaces). Lower is better.
              </p>
            </div>

            <div>
              <div className="flex items-baseline gap-3 mb-1">
                <h3 className="text-xl font-semibold text-white">Consistency</h3>
                <span className="text-sm text-gray-600 font-mono italic">/kənˈsɪstənsi/</span>
              </div>
              <p className="text-gray-400 text-lg leading-relaxed">
                Derived from the Coefficient of Variation (CV) of your IKIs. A consistency of 100 means robotic, perfect rhythm. 
                Human experts typically range from 60-80.
              </p>
            </div>
          </div>
        </section>

        {/* Tips */}
        <section className="bg-gray-900 rounded-xl border border-gray-800 p-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-amber-500/10 rounded-lg">
              <Target className="text-amber-400" size={24} />
            </div>
            <h2 className="text-2xl font-bold text-white">Improvement Strategy</h2>
          </div>

          <ul className="space-y-6 text-gray-400 text-lg">
            <li className="flex items-start gap-3">
              <span className="text-amber-400 font-bold text-lg">•</span>
              <span><strong>Accuracy is speed in waiting.</strong> Never rush beyond your ability to type accurately. Corrections are expensive.</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-amber-400 font-bold text-lg">•</span>
              <span><strong>Find your flow.</strong> Rhythm is easier to maintain than bursts of speed. Try to keep a steady metronome in your head.</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-amber-400 font-bold text-lg">•</span>
              <span><strong>Look at the rollover.</strong> If your speed has plateaued, check your rollover stat. You may need to learn to press keys simultaneously to break through.</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-amber-400 font-bold text-lg">•</span>
              <span><strong>Short, frequent sessions.</strong> 15 minutes daily is superior to a 2-hour binge once a week. Motor learning happens during sleep.</span>
            </li>
          </ul>
        </section>

      </div>
    </div>
  );
};

export default WikiPage;