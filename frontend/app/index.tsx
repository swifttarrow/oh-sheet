import { useState, useEffect, useCallback } from "react";
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Platform,
  RefreshControl,
} from "react-native";
import UploadZone from "../components/UploadZone";
import JobCard from "../components/JobCard";
import { uploadMidi, submitJob, listJobs, type JobSummary } from "../lib/api";

export default function HomeScreen() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchJobs = useCallback(async () => {
    try {
      const data = await listJobs();
      setJobs(data);
    } catch {
      // Silently fail — jobs list is secondary
    }
  }, []);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchJobs();
    setRefreshing(false);
  }, [fetchJobs]);

  const handleFileSelected = async (file: {
    uri: string;
    name: string;
    size?: number;
  }) => {
    setUploading(true);
    setError(null);

    try {
      // On web, we can create a File from the URI
      // On native, we need to fetch the URI first
      let fileObj: File;
      if (Platform.OS === "web") {
        const res = await fetch(file.uri);
        const blob = await res.blob();
        fileObj = new File([blob], file.name, { type: "audio/midi" });
      } else {
        const res = await fetch(file.uri);
        const blob = await res.blob();
        fileObj = new File([blob], file.name, { type: "audio/midi" });
      }

      // Step 1: Upload MIDI file
      const midiRef = await uploadMidi(fileObj);

      // Step 2: Submit pipeline job
      const title = file.name.replace(/\.(mid|midi)$/i, "");
      await submitJob({ midi: midiRef, title });

      // Step 3: Refresh job list
      await fetchJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#7c3aed" />
      }
    >
      {/* Hero */}
      <View style={styles.hero}>
        <Text style={styles.heroTitle}>
          Oh <Text style={styles.accent}>Sheet</Text>
        </Text>
        <Text style={styles.heroSubtitle}>
          Upload a MIDI file and we'll turn it into piano sheet music.
          Transcribed, arranged, humanized, and engraved — automatically.
        </Text>
      </View>

      {/* Upload */}
      <UploadZone
        onFileSelected={handleFileSelected}
        uploading={uploading}
        error={error}
      />

      {/* Recent Jobs */}
      {jobs.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>RECENT JOBS</Text>
          {jobs.map((job) => (
            <JobCard key={job.job_id} job={job} />
          ))}
        </View>
      )}

      {jobs.length === 0 && !uploading && (
        <View style={styles.empty}>
          <Text style={styles.emptyText}>
            No jobs yet. Upload a MIDI file to get started.
          </Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0a0a0f",
  },
  content: {
    padding: 24,
    maxWidth: 600,
    width: "100%",
    alignSelf: "center",
  },
  hero: {
    alignItems: "center",
    marginBottom: 32,
    marginTop: 40,
  },
  heroTitle: {
    fontSize: 36,
    fontWeight: "700",
    color: "#e8e8f0",
    marginBottom: 12,
  },
  accent: {
    color: "#7c3aed",
  },
  heroSubtitle: {
    fontSize: 14,
    color: "#7a7a8e",
    textAlign: "center",
    lineHeight: 22,
    maxWidth: 400,
  },
  section: {
    marginTop: 32,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: "600",
    color: "#7a7a8e",
    letterSpacing: 2,
    marginBottom: 12,
  },
  empty: {
    marginTop: 48,
    alignItems: "center",
  },
  emptyText: {
    fontSize: 13,
    color: "#7a7a8e",
    textAlign: "center",
  },
});
