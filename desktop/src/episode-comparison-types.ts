import type {DevelopmentEpisode} from './episode-types';
export interface EpisodeComparisonRequest {caseHash:string;episodeId:string}
export interface EpisodeComparisonResult {
 caseHash:string;actorEpisodeId:string;actorStrategySeal:string;pairSeal:string;
 projectionHash:string;initialProjectedObservationHash:string;
 companion:{episode:DevelopmentEpisode;episodeCanonicalJson:string};
}
export type EpisodeComparisonArm='actor'|'search';
