import { View, Text, StyleSheet, Pressable } from "react-native";
import type { JobSummary } from "../lib/api";

interface JobCardProps {
  job: JobSummary;
  onPress?: () => void;
}

const STATUS_COLORS: Record<string, string> = {
  pending: "#fbbf24",
  running: "#a78bfa",
  succeeded: "#34d399",
  failed: "#f87171",
  cancelled: "#7a7a8e",
};

export default function JobCard({ job, onPress }: JobCardProps) {
  const statusColor = STATUS_COLORS[job.status] || "#7a7a8e";

  return (
    <Pressable
      style={({ pressed }) => [styles.card, pressed && styles.cardPressed]}
      onPress={onPress}
      testID={`job-card-${job.job_id}`}
    >
      <View style={styles.row}>
        <View style={styles.info}>
          <Text style={styles.title} numberOfLines={1}>
            {job.title || "Untitled"}
          </Text>
          <Text style={styles.meta}>
            {job.variant} {job.artist ? `\u00b7 ${job.artist}` : ""}
          </Text>
        </View>
        <View style={[styles.badge, { backgroundColor: statusColor + "20" }]}>
          <Text style={[styles.badgeText, { color: statusColor }]}>
            {job.status}
          </Text>
        </View>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#12121a",
    borderWidth: 1,
    borderColor: "#2a2a3a",
    borderRadius: 8,
    padding: 14,
    marginBottom: 8,
  },
  cardPressed: {
    borderColor: "#7c3aed",
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  info: { flex: 1, marginRight: 12 },
  title: {
    fontSize: 14,
    fontWeight: "500",
    color: "#e8e8f0",
    marginBottom: 2,
  },
  meta: {
    fontSize: 12,
    color: "#7a7a8e",
  },
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 4,
  },
  badgeText: {
    fontSize: 11,
    fontWeight: "600",
    textTransform: "capitalize",
  },
});
