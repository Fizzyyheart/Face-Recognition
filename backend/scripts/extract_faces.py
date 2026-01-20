"""
视频抽脸脚本：从课堂视频中提取高质量人脸样本
用法：
    conda activate face-rec
    # 处理单个视频
    python -m backend.scripts.extract_faces --video VIdeo/VID_20251125_092626.mp4
    # 处理所有视频 + 聚类 + 匹配种子库
    python -m backend.scripts.extract_faces --all --cluster --match-seeds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import hdbscan
from sklearn.preprocessing import normalize

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings
from backend.app.face.detector import detect_faces, FaceInfo, compute_similarity
from backend.app.face.quality import assess_quality, QualityResult
from backend.app.face.tracker import SimpleTracker, Track
from backend.app.db.database import init_db, get_db
from backend.app.db.models import FaceSample, Cluster, Person, ClusterStatus


def extract_faces_from_video(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = None,
    frame_scale: float = None,
    save_images: bool = True,
    save_to_db: bool = True,
    max_frames: int = None,
) -> dict:
    """
    从视频中抽取人脸样本

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录（存放人脸截图）
        frame_interval: 每隔多少帧检测一次
        frame_scale: 帧缩放比例
        save_images: 是否保存人脸截图
        save_to_db: 是否保存到数据库
        max_frames: 最大处理帧数（调试用）

    Returns:
        统计信息 dict
    """
    frame_interval = frame_interval or settings.FRAME_SAMPLE_INTERVAL
    frame_scale = frame_scale or settings.FRAME_SCALE

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {total_frames} 帧, {fps:.1f} FPS, {width}x{height}")
    print(f"抽帧间隔: 每 {frame_interval} 帧, 缩放: {frame_scale}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化跟踪器
    tracker = SimpleTracker()

    # 统计
    stats = {
        "video": video_path.name,
        "total_frames": total_frames,
        "processed_frames": 0,
        "detected_faces": 0,
        "passed_quality": 0,
        "saved_samples": 0,
        "tracks": 0,
    }

    frame_idx = 0
    processed = 0

    # 如果要存数据库，先初始化
    if save_to_db:
        init_db()

    print("开始处理视频...")
    print(
        f"预计处理帧数: {min(total_frames, max_frames) if max_frames else total_frames}"
    )
    import sys

    sys.stdout.flush()

    pbar = tqdm(
        total=min(total_frames, max_frames) if max_frames else total_frames,
        desc="抽取人脸",
        ncols=100,
        dynamic_ncols=False,
        mininterval=0.5,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_idx >= max_frames:
            break

        # 每隔 N 帧处理一次
        if frame_idx % frame_interval == 0:
            # 缩放
            if frame_scale != 1.0:
                frame_resized = cv2.resize(
                    frame,
                    None,
                    fx=frame_scale,
                    fy=frame_scale,
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                frame_resized = frame

            # 检测人脸
            faces = detect_faces(frame_resized, extract_embedding=True)
            stats["detected_faces"] += len(faces)

            # 质量评估
            qualities = [assess_quality(frame_resized, f) for f in faces]
            passed = sum(1 for q in qualities if q.passed)
            stats["passed_quality"] += passed

            # 更新跟踪器
            tracker.update(frame_idx, faces, qualities, frame_resized)

            processed += 1

            # 实时更新进度信息
            pbar.set_postfix(
                {
                    "faces": stats["detected_faces"],
                    "passed": stats["passed_quality"],
                    "tracks": len(tracker.get_active_tracks()),
                }
            )

        frame_idx += 1
        pbar.update(1)
        pbar.refresh()

    pbar.close()
    cap.release()

    stats["processed_frames"] = processed

    # 处理所有 tracks，保存最优样本
    all_tracks = tracker.get_all_tracks()
    stats["tracks"] = len(all_tracks)

    print(f"\n共 {len(all_tracks)} 个 tracks，开始保存样本...")

    saved_count = 0
    db_samples = []

    for track in tqdm(all_tracks, desc="保存样本"):
        if track.sample_count == 0:
            continue

        # 获取最优样本
        best_samples = track.get_best_samples(k=settings.TRACK_MAX_SAMPLES)

        sample_idx = 0
        for sample in best_samples:
            # 跳过没有有效裁剪的样本（被边界截断的）
            if sample.image_crop is None:
                continue

            # 保存图片
            if save_images:
                img_name = f"track{track.track_id:04d}_sample{sample_idx:02d}.jpg"
                img_path = output_dir / img_name
                cv2.imwrite(str(img_path), sample.image_crop)
            else:
                img_path = None

            # 准备数据库记录
            if save_to_db and sample.face.embedding is not None:
                db_sample = FaceSample(
                    source_video=str(video_path),
                    frame_index=sample.frame_idx,
                    track_id=track.track_id,
                    image_path=str(img_path) if img_path else None,
                    bbox_json=json.dumps(sample.face.bbox.tolist()),
                    embedding=sample.face.embedding.astype(np.float32).tobytes(),
                    quality_score=sample.quality.quality_score,
                    blur_score=sample.quality.blur_score,
                    det_score=sample.face.det_score,
                )
                db_samples.append(db_sample)

            saved_count += 1
            sample_idx += 1

    # 批量写入数据库
    if save_to_db and db_samples:
        with get_db() as db:
            db.add_all(db_samples)
        print(f"已写入 {len(db_samples)} 条样本到数据库")

    stats["saved_samples"] = saved_count

    print("\n=== 抽取完成 ===")
    print(f"处理帧数: {stats['processed_frames']}")
    print(f"检测人脸: {stats['detected_faces']}")
    print(f"通过质量: {stats['passed_quality']}")
    print(f"跟踪轨迹: {stats['tracks']}")
    print(f"保存样本: {stats['saved_samples']}")

    return stats


def extract_all_videos(
    video_dir: Path = None,
    frame_interval: int = None,
    frame_scale: float = None,
    max_frames: int = None,
) -> list:
    """
    批量处理目录下所有视频文件

    Args:
        video_dir: 视频目录
        frame_interval: 抽帧间隔
        frame_scale: 帧缩放比例
        max_frames: 最大处理帧数（调试用）

    Returns:
        所有视频的统计信息列表
    """
    video_dir = video_dir or settings.VIDEO_DIR
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    # 查找所有视频文件
    video_files = [
        f
        for f in video_dir.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"未找到视频文件: {video_dir}")
        return []

    print(f"\n=== 找到 {len(video_files)} 个视频文件 ===")
    for v in video_files:
        print(f"  - {v.name}")

    all_stats = []
    for idx, video_path in enumerate(video_files):
        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(video_files)}] 处理视频: {video_path.name}")
        print("=" * 60)

        output_dir = settings.DATA_DIR / "faces" / video_path.stem

        try:
            stats = extract_faces_from_video(
                video_path=video_path,
                output_dir=output_dir,
                frame_interval=frame_interval,
                frame_scale=frame_scale,
                save_images=True,
                save_to_db=True,
                max_frames=max_frames,
            )
            all_stats.append(stats)

            # 保存单个视频的统计
            stats_path = output_dir / "stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"处理视频失败: {video_path.name}, 错误: {e}")
            import traceback

            traceback.print_exc()

    return all_stats


def load_seed_embeddings(seed_dir: Path = None) -> dict:
    """
    加载种子库中的人脸 embedding

    Args:
        seed_dir: 种子图片目录

    Returns:
        dict: {name: embedding}
    """
    seed_dir = seed_dir or (settings.DATA_DIR / "seeds")

    if not seed_dir.exists():
        print(f"种子目录不存在: {seed_dir}")
        return {}

    seed_embeddings = {}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    seed_files = [
        f
        for f in seed_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    print(f"\n=== 加载种子库 ({len(seed_files)} 张图片) ===")

    for img_path in tqdm(seed_files, desc="提取种子特征"):
        name = img_path.stem  # 文件名作为人名

        # 使用 numpy 读取来解决中文路径问题
        try:
            img_array = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"  警告: 读取图片失败 {img_path.name}: {e}")
            continue

        if img is None:
            print(f"  警告: 无法解码图片 {img_path.name}")
            continue

        # 检测人脸并提取 embedding
        faces = detect_faces(img, extract_embedding=True, max_faces=1)
        if not faces:
            print(f"  警告: 未检测到人脸 {img_path.name}")
            continue

        if faces[0].embedding is None:
            print(f"  警告: 无法提取特征 {img_path.name}")
            continue

        seed_embeddings[name] = faces[0].embedding
        # print(f"  已加载: {name}")

    print(f"成功加载 {len(seed_embeddings)} 个种子人脸")
    return seed_embeddings


def cluster_and_match(
    min_cluster_size: int = None,
    min_samples: int = None,
    match_threshold: float = None,
):
    """
    对数据库中的人脸样本进行聚类，并与种子库匹配

    Args:
        min_cluster_size: HDBSCAN 最小簇大小
        min_samples: HDBSCAN 最小样本数
        match_threshold: 种子匹配的相似度阈值
    """
    min_cluster_size = min_cluster_size or settings.CLUSTER_MIN_SIZE
    min_samples = min_samples or settings.CLUSTER_MIN_SAMPLES
    match_threshold = match_threshold or settings.RECOGNITION_THRESHOLD

    init_db()

    # 1. 加载所有人脸样本的 embedding
    print("\n=== 步骤 1: 加载人脸样本 ===")
    with get_db() as db:
        samples = db.query(FaceSample).filter(FaceSample.embedding.isnot(None)).all()

        if not samples:
            print("数据库中没有人脸样本")
            return

        print(f"共 {len(samples)} 个人脸样本")

        # 提取 embedding
        sample_ids = []
        embeddings = []
        for s in samples:
            emb = np.frombuffer(s.embedding, dtype=np.float32)
            if len(emb) == 512:
                sample_ids.append(s.id)
                embeddings.append(emb)

    embeddings = np.array(embeddings)
    print(f"有效 embedding: {len(embeddings)} 个")

    # 2. HDBSCAN 聚类
    print("\n=== 步骤 2: HDBSCAN 聚类 ===")
    print(f"参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    # L2 归一化后使用欧氏距离（等价于余弦距离）
    embeddings_norm = normalize(embeddings, axis=1)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    cluster_labels = clusterer.fit_predict(embeddings_norm)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    print(f"聚类结果: {n_clusters} 个簇, {n_noise} 个噪声点")

    # 3. 加载种子库
    print("\n=== 步骤 3: 加载种子库 ===")
    seed_embeddings = load_seed_embeddings()

    if not seed_embeddings:
        print("警告: 种子库为空，将跳过匹配步骤")

    # 4. 创建 Cluster 和 Person 记录
    print("\n=== 步骤 4: 创建聚类和人员记录 ===")

    with get_db() as db:
        # 清理旧的聚类数据
        db.query(Cluster).delete()
        db.query(Person).delete()
        # 重置 FaceSample 的关联
        db.query(FaceSample).update(
            {FaceSample.cluster_id: None, FaceSample.person_id: None}
        )
        db.commit()

        cluster_stats = []

        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id == -1:
                continue  # 跳过噪声

            # 获取该簇的样本索引
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_sample_ids = [sample_ids[i] for i in cluster_indices]
            cluster_embs = embeddings[cluster_mask]

            # 计算簇中心
            center_emb = cluster_embs.mean(axis=0)
            center_emb_norm = center_emb / np.linalg.norm(center_emb)

            # 与种子库匹配
            best_match_name = None
            best_match_score = 0.0

            if seed_embeddings:
                for name, seed_emb in seed_embeddings.items():
                    seed_emb_norm = seed_emb / np.linalg.norm(seed_emb)
                    similarity = float(np.dot(center_emb_norm, seed_emb_norm))
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_name = name

            # 判断是否匹配成功
            matched = best_match_score >= match_threshold

            # 创建或获取 Person
            person = None
            if matched and best_match_name:
                # 检查是否已存在该人
                person = db.query(Person).filter(Person.name == best_match_name).first()
                if not person:
                    person = Person(name=best_match_name)
                    db.add(person)
                    db.flush()

            # 获取该簇的第一个样本作为预览图
            first_sample = (
                db.query(FaceSample)
                .filter(FaceSample.id == cluster_sample_ids[0])
                .first()
            )
            preview_path = first_sample.image_path if first_sample else None

            # 创建 Cluster
            cluster = Cluster(
                status=ClusterStatus.LABELED.value
                if matched
                else ClusterStatus.PENDING.value,
                person_id=person.id if person else None,
                preview_image_path=preview_path,
                sample_count=len(cluster_sample_ids),
                center_embedding=center_emb.astype(np.float32).tobytes(),
            )
            db.add(cluster)
            db.flush()

            # 更新该簇所有样本的 cluster_id 和 person_id
            db.query(FaceSample).filter(FaceSample.id.in_(cluster_sample_ids)).update(
                {
                    FaceSample.cluster_id: cluster.id,
                    FaceSample.person_id: person.id if person else None,
                },
                synchronize_session=False,
            )

            cluster_stats.append(
                {
                    "cluster_id": cluster.id,
                    "sample_count": len(cluster_sample_ids),
                    "matched": matched,
                    "person_name": best_match_name if matched else None,
                    "match_score": round(best_match_score, 4),
                }
            )

        db.commit()

    # 5. 打印统计
    print("\n=== 聚类与匹配完成 ===")
    print(f"总簇数: {len(cluster_stats)}")

    matched_clusters = [c for c in cluster_stats if c["matched"]]
    unmatched_clusters = [c for c in cluster_stats if not c["matched"]]

    print(f"已命名: {len(matched_clusters)} 个簇")
    print(f"未命名: {len(unmatched_clusters)} 个簇")
    print(f"噪声样本: {n_noise} 个")

    if matched_clusters:
        print("\n已识别的人员:")
        for c in sorted(matched_clusters, key=lambda x: -x["sample_count"]):
            print(
                f"  - {c['person_name']}: {c['sample_count']} 个样本 (相似度: {c['match_score']:.3f})"
            )

    if unmatched_clusters:
        print("\n未识别的簇:")
        for c in sorted(unmatched_clusters, key=lambda x: -x["sample_count"])[:10]:
            print(
                f"  - 簇 {c['cluster_id']}: {c['sample_count']} 个样本 (最高相似度: {c['match_score']:.3f})"
            )

    return cluster_stats


def main():
    parser = argparse.ArgumentParser(
        description="从视频中抽取人脸样本、聚类并匹配种子库"
    )

    # 视频处理参数
    parser.add_argument("--video", type=str, default=None, help="单个视频文件路径")
    parser.add_argument("--all", action="store_true", help="处理所有视频")
    parser.add_argument(
        "--output", type=str, default=None, help="输出目录（默认 data/faces/视频名）"
    )
    parser.add_argument("--interval", type=int, default=None, help="抽帧间隔")
    parser.add_argument("--scale", type=float, default=None, help="帧缩放比例")
    parser.add_argument("--no-images", action="store_true", help="不保存人脸截图")
    parser.add_argument("--no-db", action="store_true", help="不保存到数据库")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="最大处理帧数（调试用）"
    )

    # 聚类和匹配参数
    parser.add_argument("--cluster", action="store_true", help="对样本进行聚类")
    parser.add_argument("--match-seeds", action="store_true", help="与种子库匹配人名")
    parser.add_argument(
        "--cluster-only", action="store_true", help="仅聚类（不提取视频）"
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=None, help="HDBSCAN 最小簇大小"
    )
    parser.add_argument(
        "--min-samples", type=int, default=None, help="HDBSCAN 最小样本数"
    )
    parser.add_argument(
        "--match-threshold", type=float, default=None, help="种子匹配阈值"
    )

    args = parser.parse_args()

    # 如果仅聚类
    if args.cluster_only:
        cluster_and_match(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            match_threshold=args.match_threshold,
        )
        return

    # 处理视频
    if args.all:
        # 批量处理所有视频
        all_stats = extract_all_videos(
            frame_interval=args.interval,
            frame_scale=args.scale,
            max_frames=args.max_frames,
        )

        print(f"\n=== 全部视频处理完成 ===")
        print(f"共处理 {len(all_stats)} 个视频")
        total_samples = sum(s.get("saved_samples", 0) for s in all_stats)
        print(f"总共保存 {total_samples} 个人脸样本")

    elif args.video:
        # 处理单个视频
        video_path = Path(args.video)
        if not video_path.exists():
            # 尝试相对于项目根目录
            video_path = PROJECT_ROOT / args.video
            if not video_path.exists():
                print(f"错误: 视频文件不存在: {args.video}")
                sys.exit(1)

        output_dir = (
            Path(args.output)
            if args.output
            else settings.DATA_DIR / "faces" / video_path.stem
        )

        stats = extract_faces_from_video(
            video_path=video_path,
            output_dir=output_dir,
            frame_interval=args.interval,
            frame_scale=args.scale,
            save_images=not args.no_images,
            save_to_db=not args.no_db,
            max_frames=args.max_frames,
        )

        # 保存统计信息
        stats_path = output_dir / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"统计信息已保存到: {stats_path}")

    else:
        if not (args.cluster or args.match_seeds):
            parser.print_help()
            print(
                "\n错误: 请指定 --video 或 --all 来处理视频，或使用 --cluster-only 仅聚类"
            )
            sys.exit(1)

    # 聚类和匹配
    if args.cluster or args.match_seeds:
        cluster_and_match(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            match_threshold=args.match_threshold,
        )


if __name__ == "__main__":
    main()
